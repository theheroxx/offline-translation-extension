mod FFN;
mod Tokenizer;
mod Embedding;
mod Attention;
mod PositionalEncoding;
mod Transformer;

use burn::prelude::*;
use burn_ndarray::NdArray;

use Tokenizer::batch::TranslationBatch;
use Tokenizer::dataset::load_dataset;
use Tokenizer::segmenter::{DocumentSegmenter, TokenCounter};
use Tokenizer::tokenizer::Tokenizer as BpeTokenizer;

use Embedding::Embedding as EmbeddingLayer;
use Transformer::transformer::Transformer as TransformerModel;

type Backend = NdArray<f32>;

struct SourceTokenCounter<'a> {
    tokenizer: &'a BpeTokenizer,
}

impl<'a> TokenCounter for SourceTokenCounter<'a> {
    fn count(&self, text: &str) -> usize {
        self.tokenizer.encode_source(text).len()
    }
}

struct TargetTokenCounter<'a> {
    tokenizer: &'a BpeTokenizer,
}

impl<'a> TokenCounter for TargetTokenCounter<'a> {
    fn count(&self, text: &str) -> usize {
        self.tokenizer.encode_target_labels(text).len()
    }
}

fn main() {
    println!("LOADING DATASET");

    let dataset = load_dataset("data")
        .expect("Failed to load dataset");

    println!("Loaded {} translation pairs", dataset.len());

    if dataset.is_empty() {
        panic!("Dataset is empty");
    }

    let source_texts: Vec<String> = dataset
        .iter()
        .map(|pair| pair.source.clone())
        .collect();

    let target_texts: Vec<String> = dataset
        .iter()
        .map(|pair| pair.target.clone())
        .collect();

    println!("Source training texts: {}", source_texts.len());
    println!("Target training texts: {}", target_texts.len());

    println!("\nCREATING SOURCE BPE TOKENIZER");

    let mut source_tokenizer = BpeTokenizer::new(100);

    println!("\nTRAINING SOURCE BPE TOKENIZER");

    source_tokenizer.train(&source_texts);

    println!(
        "Source vocabulary size: {}",
        source_tokenizer.vocab_size()
    );

    println!("\nCREATING TARGET BPE TOKENIZER");

    let mut target_tokenizer = BpeTokenizer::new(100);

    println!("\nTRAINING TARGET BPE TOKENIZER");

    target_tokenizer.train(&target_texts);

    println!(
        "Target vocabulary size: {}",
        target_tokenizer.vocab_size()
    );

    let max_encoder_sequence_length = 512;
    let max_decoder_sequence_length = 512;

    println!("\n================================================");
    println!("DOCUMENT SEGMENTATION");
    println!("================================================");

    let source_counter = SourceTokenCounter {
        tokenizer: &source_tokenizer,
    };

    let target_counter = TargetTokenCounter {
        tokenizer: &target_tokenizer,
    };

    let segmenter = DocumentSegmenter::new(
        max_encoder_sequence_length,
        max_decoder_sequence_length,
        source_counter,
        target_counter,
    );

    let segments = segmenter.segment_dataset(&dataset);

    println!(
        "Original translation pairs: {}",
        dataset.len()
    );

    println!(
        "Generated translation segments: {}",
        segments.len()
    );

    if segments.is_empty() {
        panic!("Segmentation produced no translation segments");
    }

    println!("\n================================================");
    println!("SEGMENT TOKEN LENGTH DIAGNOSTICS");
    println!("================================================");

    let mut max_segment_source_tokens = 0usize;
    let mut max_segment_target_tokens = 0usize;

    let mut min_segment_source_tokens = usize::MAX;
    let mut min_segment_target_tokens = usize::MAX;

    let mut total_segment_source_tokens = 0usize;
    let mut total_segment_target_tokens = 0usize;

    let mut source_segments_over_limit = 0usize;
    let mut target_segments_over_limit = 0usize;

    for (index, segment) in segments.iter().enumerate() {
        let source_tokens = source_tokenizer
            .encode_source(&segment.source)
            .len();

        let target_tokens = target_tokenizer
            .encode_target_labels(&segment.target)
            .len();

        max_segment_source_tokens =
            max_segment_source_tokens.max(source_tokens);

        max_segment_target_tokens =
            max_segment_target_tokens.max(target_tokens);

        min_segment_source_tokens =
            min_segment_source_tokens.min(source_tokens);

        min_segment_target_tokens =
            min_segment_target_tokens.min(target_tokens);

        total_segment_source_tokens += source_tokens;
        total_segment_target_tokens += target_tokens;

        if source_tokens > max_encoder_sequence_length {
            source_segments_over_limit += 1;
        }

        if target_tokens > max_decoder_sequence_length {
            target_segments_over_limit += 1;
        }

        if index < 20 {
            println!(
                "Segment {:>4}: source={:>4} tokens | target={:>4} tokens",
                index + 1,
                source_tokens,
                target_tokens
            );
        }
    }

    let segment_count = segments.len();

    let average_segment_source_tokens =
        total_segment_source_tokens as f64 / segment_count as f64;

    let average_segment_target_tokens =
        total_segment_target_tokens as f64 / segment_count as f64;

    println!("\n================================================");
    println!("SEGMENT TOKEN SUMMARY");
    println!("================================================");

    println!("Segment count: {}", segment_count);

    println!(
        "Minimum source tokens: {}",
        min_segment_source_tokens
    );

    println!(
        "Maximum source tokens: {}",
        max_segment_source_tokens
    );

    println!(
        "Average source tokens: {:.2}",
        average_segment_source_tokens
    );

    println!(
        "Minimum target tokens: {}",
        min_segment_target_tokens
    );

    println!(
        "Maximum target tokens: {}",
        max_segment_target_tokens
    );

    println!(
        "Average target tokens: {:.2}",
        average_segment_target_tokens
    );

    println!(
        "Source segments over {} tokens: {}",
        max_encoder_sequence_length,
        source_segments_over_limit
    );

    println!(
        "Target segments over {} tokens: {}",
        max_decoder_sequence_length,
        target_segments_over_limit
    );

    assert_eq!(
        source_segments_over_limit,
        0,
        "Some source segments exceed the encoder context length"
    );

    assert_eq!(
        target_segments_over_limit,
        0,
        "Some target segments exceed the decoder context length"
    );

    println!("\n================================================");
    println!("CREATING TRANSLATION BATCH");
    println!("================================================");

    let batch_size = segments.len().min(8);

    let batch = TranslationBatch::from_segments(
        &segments[..batch_size],
        &source_tokenizer,
        &target_tokenizer,
    );

    batch.assert_within_context(
        max_encoder_sequence_length,
        max_decoder_sequence_length,
    );

    println!("Batch size: {}", batch.batch_size());
    println!("Source shape: {:?}", batch.source_shape());
    println!("Target shape: {:?}", batch.target_shape());

    println!(
        "Maximum source length: {}",
        batch.max_source_length
    );

    println!(
        "Maximum target length: {}",
        batch.max_target_length
    );

    let device = Default::default();

    println!("\n================================================");
    println!("CREATING SOURCE EMBEDDING");
    println!("================================================");

    let source_embedding_dimension = 128;

    let source_embedding = EmbeddingLayer::<Backend>::new(
        source_tokenizer.vocab_size(),
        source_embedding_dimension,
        &device,
    );

    println!(
        "Source vocabulary size: {}",
        source_tokenizer.vocab_size()
    );

    println!(
        "Source embedding dimension: {}",
        source_embedding_dimension
    );

    println!("\n================================================");
    println!("CREATING TARGET EMBEDDING");
    println!("================================================");

    let target_embedding_dimension = 128;

    let target_embedding = EmbeddingLayer::<Backend>::new(
        target_tokenizer.vocab_size(),
        target_embedding_dimension,
        &device,
    );

    println!(
        "Target vocabulary size: {}",
        target_tokenizer.vocab_size()
    );

    println!(
        "Target embedding dimension: {}",
        target_embedding_dimension
    );

    assert_eq!(
        source_embedding_dimension,
        target_embedding_dimension,
        "Source and target embedding dimensions must match"
    );

    println!("\n================================================");
    println!("CREATING TRANSFORMER");
    println!("================================================");

    let transformer = TransformerModel::<Backend>::new(
        6,
        6,
        source_embedding_dimension,
        8,
        512,
        max_encoder_sequence_length,
        max_decoder_sequence_length,
        target_tokenizer.vocab_size(),
        &device,
    );

    println!(
        "Encoder layers: {}",
        transformer.num_encoder_layers
    );

    println!(
        "Decoder layers: {}",
        transformer.num_decoder_layers
    );

    println!(
        "Embedding dimension: {}",
        transformer.embedding_dimension
    );

    println!(
        "Attention heads: {}",
        transformer.num_heads
    );

    println!(
        "FFN hidden dimension: {}",
        transformer.ffn_hidden_dimension
    );

    println!(
        "Maximum encoder sequence length: {}",
        transformer.max_encoder_sequence_length
    );

    println!(
        "Maximum decoder sequence length: {}",
        transformer.max_decoder_sequence_length
    );

    println!(
        "Target vocabulary size: {}",
        transformer.target_vocab_size
    );

    println!("\n================================================");
    println!("CREATING INPUT TENSORS");
    println!("================================================");

    let source_ids = batch.source_ids_flattened();
    let target_input_ids = batch.target_input_ids_flattened();

    let source_ids_i64: Vec<i64> = source_ids
        .iter()
        .map(|&id| id as i64)
        .collect();

    let target_input_ids_i64: Vec<i64> = target_input_ids
        .iter()
        .map(|&id| id as i64)
        .collect();

    let source_tensor =
        Tensor::<Backend, 1, Int>::from_ints(
            source_ids_i64.as_slice(),
            &device,
        )
        .reshape([
            batch.batch_size(),
            batch.max_source_length,
        ]);

    let target_tensor =
        Tensor::<Backend, 1, Int>::from_ints(
            target_input_ids_i64.as_slice(),
            &device,
        )
        .reshape([
            batch.batch_size(),
            batch.max_target_length,
        ]);

    println!(
        "Source tensor shape: {:?}",
        source_tensor.shape()
    );

    println!(
        "Target tensor shape: {:?}",
        target_tensor.shape()
    );

    println!("\n================================================");
    println!("CREATING PADDING MASKS");
    println!("================================================");

    let source_padding_values = batch
        .source_padding_mask()
        .into_iter()
        .flatten()
        .collect::<Vec<bool>>();

    let target_padding_values = batch
        .target_padding_mask()
        .into_iter()
        .flatten()
        .collect::<Vec<bool>>();

    let source_padding_mask =
        Tensor::<Backend, 1, Bool>::from_bool(
            source_padding_values.as_slice().into(),
            &device,
        )
        .reshape([
            batch.batch_size(),
            batch.max_source_length,
        ]);

    let target_padding_mask =
        Tensor::<Backend, 1, Bool>::from_bool(
            target_padding_values.as_slice().into(),
            &device,
        )
        .reshape([
            batch.batch_size(),
            batch.max_target_length,
        ]);

    println!(
        "Source padding mask shape: {:?}",
        source_padding_mask.shape()
    );

    println!(
        "Target padding mask shape: {:?}",
        target_padding_mask.shape()
    );

    println!("\n================================================");
    println!("CREATING EMBEDDINGS");
    println!("================================================");

    let source_embeddings =
        source_embedding.forward(source_tensor);

    let target_embeddings =
        target_embedding.forward(target_tensor);

    println!(
        "Source embeddings shape: {:?}",
        source_embeddings.shape()
    );

    println!(
        "Target embeddings shape: {:?}",
        target_embeddings.shape()
    );

    println!("\n================================================");
    println!("RUNNING TRANSFORMER");
    println!("================================================");

    let output = transformer.forward(
        source_embeddings,
        target_embeddings,
        Some(source_padding_mask),
        Some(target_padding_mask),
    );

    println!(
        "Transformer output shape: {:?}",
        output.shape()
    );

    println!("\n================================================");
    println!("PIPELINE COMPLETE");
    println!("================================================");
}
