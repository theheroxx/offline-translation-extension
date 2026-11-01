mod FFN;
mod Tokenizer;
mod Embedding;
mod Attention;
mod PositionalEncoding;
mod Transformer;

use std::error::Error;
use std::path::Path;
use std::time::Instant;

use burn::backend::Autodiff;
use burn::module::Module;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::CompactRecorder;
use burn::tensor::backend::Backend as BurnBackend;
use burn_cuda::{Cuda, CudaDevice};

use crate::Embedding::embedding::Embedding as TokenEmbedding;
use crate::Tokenizer::batch::TranslationBatch;
use crate::Tokenizer::dataset::{load_dataset, TranslationPair};
use crate::Tokenizer::segmenter::{
    BpeTokenCounter,
    DocumentSegmenter,
    TranslationSegment,
};
use crate::Tokenizer::tokenizer::Tokenizer as BPETokenizer;
use crate::Transformer::transformer::Transformer as TransformerModel;

type InnerBackend = Cuda<f32, i32>;
type TrainBackend = Autodiff<InnerBackend>;

const DATA_DIR: &str = "data";

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

const TOKENIZER_MERGES: usize = 100;

const MODEL_DIR: &str = "model";
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
        let source_embedding = TokenEmbedding::new(
            source_vocab_size,
            embedding_dimension,
            device,
        );

        let target_embedding = TokenEmbedding::new(
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

fn tensor_scalar<B: BurnBackend>(
    tensor: Tensor<B, 1>,
) -> f64 {
    tensor
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .expect("Expected a scalar tensor")[0] as f64
}

fn valid_target_tokens(batch: &TranslationBatch) -> usize {
    batch.target_lengths.iter().sum()
}

fn create_batches(
    segments: &[TranslationSegment],
    source_tokenizer: &BPETokenizer,
    target_tokenizer: &BPETokenizer,
    batch_size: usize,
) -> Vec<TranslationBatch> {
    assert!(batch_size > 0);

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

fn shuffle<T>(
    items: &mut [T],
    seed: u64,
) {
    let mut state = seed.max(1);

    for i in (1..items.len()).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;

        let j = (state as usize) % (i + 1);
        items.swap(i, j);
    }
}

fn split_translation_pairs(
    dataset: &[TranslationPair],
    validation_ratio: f32,
) -> (Vec<TranslationPair>, Vec<TranslationPair>) {
    assert!(
        validation_ratio > 0.0 && validation_ratio < 1.0,
        "Validation ratio must be between zero and one"
    );

    assert!(
        dataset.len() >= 2,
        "At least two translation pairs are required"
    );

    let mut shuffled = dataset.to_vec();
    shuffle(&mut shuffled, 0xA51C_92D7);

    let validation_size =
        ((shuffled.len() as f32) * validation_ratio).round() as usize;

    let validation_size =
        validation_size.clamp(1, shuffled.len() - 1);

    let split_index = shuffled.len() - validation_size;

    let train = shuffled[..split_index].to_vec();
    let validation = shuffled[split_index..].to_vec();

    (train, validation)
}

fn train_tokenizers(
    training_pairs: &[TranslationPair],
) -> (BPETokenizer, BPETokenizer) {
    let source_texts = training_pairs
        .iter()
        .map(|pair| pair.source.clone())
        .collect::<Vec<_>>();

    let target_texts = training_pairs
        .iter()
        .map(|pair| pair.target.clone())
        .collect::<Vec<_>>();

    let mut source_tokenizer =
        BPETokenizer::new(TOKENIZER_MERGES);

    let mut target_tokenizer =
        BPETokenizer::new(TOKENIZER_MERGES);

    source_tokenizer.train(&source_texts);
    target_tokenizer.train(&target_texts);

    source_tokenizer
        .validate()
        .expect("Source tokenizer validation failed");

    target_tokenizer
        .validate()
        .expect("Target tokenizer validation failed");

    (source_tokenizer, target_tokenizer)
}

fn segment_dataset(
    training_pairs: &[TranslationPair],
    validation_pairs: &[TranslationPair],
    source_tokenizer: &BPETokenizer,
    target_tokenizer: &BPETokenizer,
) -> Result<
    (Vec<TranslationSegment>, Vec<TranslationSegment>),
    String,
> {
    let segmenter = DocumentSegmenter::new(
        MAX_SOURCE_TOKENS,
        MAX_TARGET_TOKENS,
        BpeTokenCounter::source(source_tokenizer),
        BpeTokenCounter::target(target_tokenizer),
    );

    let training_segments =
        segmenter.segment_dataset_checked(training_pairs)?;

    let validation_segments =
        segmenter.segment_dataset_checked(validation_pairs)?;

    Ok((training_segments, validation_segments))
}

fn evaluate<B: BurnBackend>(
    model: &TranslationModel<B>,
    segments: &[TranslationSegment],
    source_tokenizer: &BPETokenizer,
    target_tokenizer: &BPETokenizer,
    batch_size: usize,
    device: &B::Device,
) -> f64 {
    assert!(
        !segments.is_empty(),
        "Validation dataset cannot be empty"
    );

    let batches = create_batches(
        segments,
        source_tokenizer,
        target_tokenizer,
        batch_size,
    );

    let mut total_weighted_loss = 0.0;
    let mut total_tokens = 0usize;

    for batch in batches {
        let (
            source_ids,
            target_input_ids,
            target_label_ids,
            source_padding_mask,
            target_padding_mask,
        ) = batch_tensors::<B>(&batch, device);

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

        let token_count = valid_target_tokens(&batch);

        if token_count == 0 {
            continue;
        }

        let loss_value = tensor_scalar(loss);

        total_weighted_loss += loss_value * token_count as f64;
        total_tokens += token_count;
    }

    assert!(
        total_tokens > 0,
        "Validation dataset contains no valid target tokens"
    );

    total_weighted_loss / total_tokens as f64
}

fn save_checkpoint<B: BurnBackend>(
    model: &TranslationModel<B>,
    path: &str,
) -> Result<(), Box<dyn Error>> {
    model
        .clone()
        .save_file(path, &CompactRecorder::new())?;

    Ok(())
}

fn save_training_config(
    source_vocab_size: usize,
    target_vocab_size: usize,
) -> Result<(), Box<dyn Error>> {
    let config = format!(
        r#"{{
  "source_vocab_size": {},
  "target_vocab_size": {},
  "embedding_dimension": {},
  "num_heads": {},
  "ffn_hidden_dimension": {},
  "num_encoder_layers": {},
  "num_decoder_layers": {},
  "max_source_tokens": {},
  "max_target_tokens": {},
  "batch_size": {},
  "epochs": {},
  "learning_rate": {},
  "weight_decay": {},
  "tokenizer_merges": {}
}}"#,
        source_vocab_size,
        target_vocab_size,
        EMBEDDING_DIMENSION,
        NUM_HEADS,
        FFN_HIDDEN_DIMENSION,
        NUM_ENCODER_LAYERS,
        NUM_DECODER_LAYERS,
        MAX_SOURCE_TOKENS,
        MAX_TARGET_TOKENS,
        BATCH_SIZE,
        EPOCHS,
        LEARNING_RATE,
        WEIGHT_DECAY,
        TOKENIZER_MERGES,
    );

    std::fs::write(
        Path::new(MODEL_DIR).join("config.json"),
        config,
    )?;

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("Starting CUDA Transformer training");

    assert!(BATCH_SIZE > 0);
    assert!(EPOCHS > 0);
    assert!(EMBEDDING_DIMENSION % NUM_HEADS == 0);

    let device = CudaDevice::default();

    println!("CUDA device initialized");

    let dataset = load_dataset(DATA_DIR)?;

    println!(
        "Loaded {} translation pairs",
        dataset.len()
    );

    if dataset.len() < 2 {
        return Err(
            "At least two translation pairs are required".into()
        );
    }

    let (training_pairs, validation_pairs) =
        split_translation_pairs(
            &dataset,
            VALIDATION_RATIO,
        );

    println!(
        "Training pairs: {}",
        training_pairs.len()
    );

    println!(
        "Validation pairs: {}",
        validation_pairs.len()
    );

    let (source_tokenizer, target_tokenizer) =
        train_tokenizers(&training_pairs);

    println!(
        "Source vocabulary size: {}",
        source_tokenizer.vocab_size()
    );

    println!(
        "Target vocabulary size: {}",
        target_tokenizer.vocab_size()
    );

    std::fs::create_dir_all(MODEL_DIR)?;
    std::fs::create_dir_all(CHECKPOINT_DIR)?;

    source_tokenizer.save_json(
        Path::new(MODEL_DIR).join("source_tokenizer.json"),
    )?;

    target_tokenizer.save_json(
        Path::new(MODEL_DIR).join("target_tokenizer.json"),
    )?;

    save_training_config(
        source_tokenizer.vocab_size(),
        target_tokenizer.vocab_size(),
    )?;

    let (
        training_segments,
        validation_segments,
    ) = segment_dataset(
        &training_pairs,
        &validation_pairs,
        &source_tokenizer,
        &target_tokenizer,
    )?;

    if training_segments.is_empty() {
        return Err("Training dataset produced no segments".into());
    }

    if validation_segments.is_empty() {
        return Err("Validation dataset produced no segments".into());
    }

    println!(
        "Training segments: {}",
        training_segments.len()
    );

    println!(
        "Validation segments: {}",
        validation_segments.len()
    );

    let model = TranslationModel::<TrainBackend>::new(
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

    let mut model = model;

    let mut optimizer = AdamWConfig::new()
        .with_weight_decay(WEIGHT_DECAY)
        .init::<
            TrainBackend,
            TranslationModel<TrainBackend>,
        >();

    println!("Model initialized on CUDA");

    let mut best_validation_loss = f64::INFINITY;

    for epoch in 1..=EPOCHS {
        let epoch_start = Instant::now();

        let mut shuffled_segments =
            training_segments.clone();

        shuffle(
            &mut shuffled_segments,
            0x7A31_2D95 ^ epoch as u64,
        );

        let train_batches = create_batches(
            &shuffled_segments,
            &source_tokenizer,
            &target_tokenizer,
            BATCH_SIZE,
        );

        let mut total_weighted_loss = 0.0;
        let mut total_tokens = 0usize;

        for (batch_index, batch) in
            train_batches.iter().enumerate()
        {
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

            let loss_value =
                tensor_scalar(loss.clone());

            let token_count =
                valid_target_tokens(batch);

            let gradients = loss.backward();

            let gradients = GradientsParams::from_grads(
                gradients,
                &model,
            );

            model = optimizer.step(
                LEARNING_RATE,
                model,
                gradients,
            );

            total_weighted_loss +=
                loss_value * token_count as f64;

            total_tokens += token_count;

            let completed_batches = batch_index + 1;

            if completed_batches % 10 == 0
                || completed_batches == train_batches.len()
            {
                let average_loss =
                    total_weighted_loss
                    / total_tokens.max(1) as f64;

                println!(
                    "Epoch {}/{} | Batch {}/{} | Loss {:.6}",
                    epoch,
                    EPOCHS,
                    completed_batches,
                    train_batches.len(),
                    average_loss,
                );
            }
        }

        if total_tokens == 0 {
            return Err(
                "Training dataset contains no valid target tokens".into()
            );
        }

        let train_loss =
            total_weighted_loss / total_tokens as f64;

        let validation_loss = evaluate(
            &model,
            &validation_segments,
            &source_tokenizer,
            &target_tokenizer,
            BATCH_SIZE,
            &device,
        );

        save_checkpoint(
            &model,
            &format!("{}/latest", CHECKPOINT_DIR),
        )?;

        let is_best =
            validation_loss < best_validation_loss;

        if is_best {
            best_validation_loss = validation_loss;

            save_checkpoint(
                &model,
                &format!("{}/best", CHECKPOINT_DIR),
            )?;
        }

        println!(
            "Epoch {}/{} complete | train_loss {:.6} | validation_loss {:.6} | best_validation_loss {:.6} | time {:.2}s",
            epoch,
            EPOCHS,
            train_loss,
            validation_loss,
            best_validation_loss,
            epoch_start.elapsed().as_secs_f64(),
        );

        if is_best {
            println!("Best model checkpoint updated");
        }
    }

    println!("Training complete");

    println!(
        "Best validation loss: {:.6}",
        best_validation_loss
    );

    println!(
        "Best checkpoint: {}/best",
        CHECKPOINT_DIR
    );

    Ok(())
}
