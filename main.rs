mod FFN;
mod Tokenizer;
mod Embedding;
mod Attention;
mod PositionalEncoding;
mod Transformer;

use std::path::Path;
use std::time::Instant;

use burn::backend::Autodiff;
use burn::module::Module;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::backend::Backend as BurnBackend;
use burn_cuda::{Cuda, CudaDevice};

use crate::Embedding::embedding::Embedding as TokenEmbedding;
use crate::Tokenizer::batch::TranslationBatch;
use crate::Tokenizer::dataset::load_dataset;
use crate::Tokenizer::segmenter::{
    DocumentSegmenter,
    TranslationSegment,
    WhitespaceTokenCounter,
};
use crate::Tokenizer::tokenizer::Tokenizer as BPETokenizer;
use crate::Transformer::transformer::Transformer as TransformerModel;

type InnerBackend = Cuda<f32, i32>;
type TrainBackend = Autodiff<InnerBackend>;

const DATA_DIR: &str = "data";

const SOURCE_VOCAB_SIZE: usize = 186;
const TARGET_VOCAB_SIZE: usize = 197;

const EMBEDDING_DIMENSION: usize = 256;
const NUM_HEADS: usize = 8;
const FFN_HIDDEN_DIMENSION: usize = 1024;

const NUM_ENCODER_LAYERS: usize = 4;
const NUM_DECODER_LAYERS: usize = 4;

const MAX_SOURCE_TOKENS: usize = 256;
const MAX_TARGET_TOKENS: usize = 256;

const BATCH_SIZE: usize = 2;
const EPOCHS: usize = 10;

const LEARNING_RATE: f64 = 0.0001;
const WEIGHT_DECAY: f32 = 0.01;

const VALIDATION_RATIO: f32 = 0.1;

const CHECKPOINT_DIR: &str = "checkpoints";

#[derive(Module, Debug)]
struct TranslationModel<B: BurnBackend> {
    source_embedding: TokenEmbedding<B>,
    target_embedding: TokenEmbedding<B>,
    transformer: TransformerModel<B>,
}

impl<B: BurnBackend> TranslationModel<B> {
    fn new(
        source_vocab_size: usize,
        target_vocab_size: usize,
        embedding_dimension: usize,
        num_encoder_layers: usize,
        num_decoder_layers: usize,
        num_heads: usize,
        ffn_hidden_dimension: usize,
        max_source_tokens: usize,
        max_target_tokens: usize,
        device: &B::Device,
    ) -> Self {
        let source_embedding =
            TokenEmbedding::new(
                source_vocab_size,
                embedding_dimension,
                device,
            );

        let target_embedding =
            TokenEmbedding::new(
                target_vocab_size,
                embedding_dimension,
                device,
            );

        let transformer = TransformerModel::new(
            num_encoder_layers,
            num_decoder_layers,
            embedding_dimension,
            num_heads,
            ffn_hidden_dimension,
            max_source_tokens,
            max_target_tokens,
            target_vocab_size,
            device,
        );

        Self {
            source_embedding,
            target_embedding,
            transformer,
        }
    }

    fn forward(
        &self,
        source_ids: Tensor<B, 2, Int>,
        target_input_ids: Tensor<B, 2, Int>,
        source_padding_mask: Tensor<B, 2, Bool>,
        target_padding_mask: Tensor<B, 2, Bool>,
    ) -> Tensor<B, 3> {
        let source_embeddings =
            self.source_embedding.forward(source_ids);

        let target_embeddings =
            self.target_embedding.forward(target_input_ids);

        self.transformer.forward(
            source_embeddings,
            target_embeddings,
            Some(source_padding_mask),
            Some(target_padding_mask),
        )
    }
}

fn create_int_tensor<B: BurnBackend>(
    data: Vec<usize>,
    shape: [usize; 2],
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    let data = data
        .into_iter()
        .map(|value| value as i64)
        .collect::<Vec<_>>();

    Tensor::<B, 2, Int>::from_data(
        TensorData::new(data, Shape::new(shape)),
        device,
    )
}

fn batch_tensors<B: BurnBackend>(
    batch: &TranslationBatch,
    device: &B::Device,
) -> (
    Tensor<B, 2, Int>,
    Tensor<B, 2, Int>,
    Tensor<B, 2, Int>,
    Tensor<B, 2, Bool>,
    Tensor<B, 2, Bool>,
) {
    let source_ids = create_int_tensor(
        batch.source_ids_flattened(),
        batch.source_shape(),
        device,
    );

    let target_input_ids = create_int_tensor(
        batch.target_input_ids_flattened(),
        batch.target_shape(),
        device,
    );

    let target_label_ids = create_int_tensor(
        batch.target_label_ids_flattened(),
        batch.target_shape(),
        device,
    );

    let source_padding_mask =
        source_ids.clone().equal_elem(batch.pad_id as i64);

    let target_padding_mask =
        target_input_ids.clone().equal_elem(batch.pad_id as i64);

    (
        source_ids,
        target_input_ids,
        target_label_ids,
        source_padding_mask,
        target_padding_mask,
    )
}

fn translation_loss<B: BurnBackend>(
    logits: Tensor<B, 3>,
    labels: Tensor<B, 2, Int>,
    pad_id: usize,
) -> Tensor<B, 1> {
    let [batch_size, sequence_length, vocab_size] =
        logits.shape().dims::<3>();

    let logits =
        logits.reshape([batch_size * sequence_length, vocab_size]);

    let labels =
        labels.reshape([batch_size * sequence_length]);

    CrossEntropyLossConfig::new()
        .with_pad_tokens(Some(vec![pad_id]))
        .init(&logits.device())
        .forward(logits, labels)
}

fn create_batches(
    segments: &[TranslationSegment],
    source_tokenizer: &BPETokenizer,
    target_tokenizer: &BPETokenizer,
    batch_size: usize,
) -> Vec<TranslationBatch> {
    segments
        .chunks(batch_size)
        .map(|chunk| {
            TranslationBatch::from_segments(
                chunk,
                source_tokenizer,
                target_tokenizer,
            )
        })
        .collect()
}

fn split_dataset(
    segments: &[TranslationSegment],
    validation_ratio: f32,
) -> (
    Vec<TranslationSegment>,
    Vec<TranslationSegment>,
) {
    let total = segments.len();

    if total < 2 {
        return (segments.to_vec(), Vec::new());
    }

    let validation_size =
        ((total as f32) * validation_ratio).round() as usize;

    let validation_size =
        validation_size.clamp(1, total - 1);

    let split_index =
        total - validation_size;

    let train =
        segments[..split_index].to_vec();

    let validation =
        segments[split_index..].to_vec();

    (train, validation)
}

fn evaluate<B: BurnBackend>(
    model: &TranslationModel<B>,
    segments: &[TranslationSegment],
    source_tokenizer: &BPETokenizer,
    target_tokenizer: &BPETokenizer,
    batch_size: usize,
    device: &B::Device,
) -> f64 {
    if segments.is_empty() {
        return 0.0;
    }

    let batches = create_batches(
        segments,
        source_tokenizer,
        target_tokenizer,
        batch_size,
    );

    let mut total_loss = 0.0;
    let mut batch_count = 0usize;

    for batch in batches {
        let (
            source_ids,
            target_input_ids,
            target_label_ids,
            source_padding_mask,
            target_padding_mask,
        ) = batch_tensors::<B>(
            &batch,
            device,
        );

        let logits = model.forward(
            source_ids,
            target_input_ids,
            source_padding_mask,
            target_padding_mask,
        );

        let loss = translation_loss(
            logits,
            target_label_ids,
            batch.pad_id,
        );

        let loss_value = loss
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .expect(
                "Validation loss tensor must contain one value",
            )[0] as f64;

        total_loss += loss_value;
        batch_count += 1;
    }

    total_loss / batch_count as f64
}

fn main() {
    println!("Starting CUDA Transformer training");

    let device = CudaDevice::default();

    println!("CUDA device initialized");

    let dataset =
        load_dataset(DATA_DIR)
            .expect("Failed to load translation dataset");

    println!(
        "Loaded {} translation pairs",
        dataset.len()
    );


    let mut source_tokenizer =
        BPETokenizer::new(100);

    let mut target_tokenizer =
        BPETokenizer::new(100);

    let source_texts = dataset
        .iter()
        .map(|pair| pair.source.clone())
        .collect::<Vec<_>>();

    let target_texts = dataset
        .iter()
        .map(|pair| pair.target.clone())
        .collect::<Vec<_>>();

    source_tokenizer.train(
        &source_texts,
    );

    target_tokenizer.train(
        &target_texts,
    );


    println!(
        "Source vocabulary size: {}",
        source_tokenizer.vocab_size()
    );

    println!(
        "Target vocabulary size: {}",
        target_tokenizer.vocab_size()
    );

    assert_eq!(
        source_tokenizer.vocab_size(),
        SOURCE_VOCAB_SIZE,
        "Source tokenizer vocabulary size does not match configured SOURCE_VOCAB_SIZE"
    );

    assert_eq!(
        target_tokenizer.vocab_size(),
        TARGET_VOCAB_SIZE,
        "Target tokenizer vocabulary size does not match configured TARGET_VOCAB_SIZE"
    );

    let segmenter =
        DocumentSegmenter::new(
            MAX_SOURCE_TOKENS,
            MAX_TARGET_TOKENS,
            WhitespaceTokenCounter,
            WhitespaceTokenCounter,
        );

    let segments =
        segmenter.segment_dataset(&dataset);

    println!(
        "Created {} training segments",
        segments.len()
    );

    let (
        train_segments,
        validation_segments,
    ) = split_dataset(
        &segments,
        VALIDATION_RATIO,
    );

    println!(
        "Training segments: {}",
        train_segments.len()
    );

    println!(
        "Validation segments: {}",
        validation_segments.len()
    );

    let train_batches =
        create_batches(
            &train_segments,
            &source_tokenizer,
            &target_tokenizer,
            BATCH_SIZE,
        );

    println!(
        "Training batches: {}",
        train_batches.len()
    );

    let max_source_length =
        train_batches
            .iter()
            .map(|batch| batch.max_source_length)
            .max()
            .unwrap_or(0);

    let max_target_length =
        train_batches
            .iter()
            .map(|batch| batch.max_target_length)
            .max()
            .unwrap_or(0);

    println!(
        "Maximum source sequence in batches: {}",
        max_source_length
    );

    println!(
        "Maximum target sequence in batches: {}",
        max_target_length
    );

    let mut model =
        TranslationModel::<TrainBackend>::new(
            source_tokenizer.vocab_size(),
            target_tokenizer.vocab_size(),
            EMBEDDING_DIMENSION,
            NUM_ENCODER_LAYERS,
            NUM_DECODER_LAYERS,
            NUM_HEADS,
            FFN_HIDDEN_DIMENSION,
            MAX_SOURCE_TOKENS,
            MAX_TARGET_TOKENS,
            &device,
        );

    let mut optimizer =
        AdamWConfig::new()
            .with_weight_decay(WEIGHT_DECAY)
            .init::<
                TrainBackend,
                TranslationModel<TrainBackend>,
            >();

    std::fs::create_dir_all(
        Path::new(CHECKPOINT_DIR),
    )
    .expect(
        "Failed to create checkpoint directory",
    );

    println!("Model initialized on CUDA");

    for epoch in 1..=EPOCHS {
        let epoch_start =
            Instant::now();

        let mut epoch_loss =
            0.0;

        let mut batch_count =
            0usize;

        for batch in &train_batches {
            let (
                source_ids,
                target_input_ids,
                target_label_ids,
                source_padding_mask,
                target_padding_mask,
            ) = batch_tensors::<TrainBackend>(
                batch,
                &device,
            );

            let logits =
                model.forward(
                    source_ids,
                    target_input_ids,
                    source_padding_mask,
                    target_padding_mask,
                );

            let loss =
                translation_loss(
                    logits,
                    target_label_ids,
                    batch.pad_id,
                );

            let loss_value =
                loss.clone()
                    .into_data()
                    .convert::<f32>()
                    .to_vec::<f32>()
                    .expect(
                        "Training loss tensor must contain one value",
                    )[0] as f64;

            let grads =
                loss.backward();

            let grads =
                GradientsParams::from_grads(
                    grads,
                    &model,
                );

            model =
                optimizer.step(
                    LEARNING_RATE,
                    model,
                    grads,
                );

            epoch_loss +=
                loss_value;

            batch_count +=
                1;

            if batch_count % 10 == 0
                || batch_count == train_batches.len()
            {
                let average_loss =
                    epoch_loss
                        / batch_count
                            .max(1)
                            as f64;

                println!(
                    "Epoch {}/{} | Batch {}/{} | Loss {:.6}",
                    epoch,
                    EPOCHS,
                    batch_count,
                    train_batches.len(),
                    average_loss
                );
            }
        }

        let train_loss =
            epoch_loss
                / batch_count.max(1)
                    as f64;

        let validation_loss =
            evaluate(
                &model,
                &validation_segments,
                &source_tokenizer,
                &target_tokenizer,
                BATCH_SIZE,
                &device,
            );

        println!(
            "Epoch {}/{} complete | train_loss {:.6} | validation_loss {:.6} | time {:.2}s",
            epoch,
            EPOCHS,
            train_loss,
            validation_loss,
            epoch_start.elapsed().as_secs_f64()
        );
    }

    println!("Training complete");
}
