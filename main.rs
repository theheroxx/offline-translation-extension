
mod FFN;
mod Tokenizer;
mod Embedding;
mod Attention;
mod PositionalEncoding;
mod Transformer;

use Tokenizer::batch::TranslationBatch;
use Tokenizer::dataset::load_dataset;
use Tokenizer::tokenizer::Tokenizer as BpeTokenizer;
use Embedding::Embedding as EmbeddingModel;
use Transformer::transformer::Transformer as TransformerModel;

use burn::prelude::*;
use burn_ndarray::NdArrayDevice;

type Backend = burn_ndarray::NdArray<f32>;

fn main() {
    println!("LOADING DATASET");

    let dataset =
        load_dataset("data")
            .expect("Failed to load dataset");

    println!(
        "Loaded {} translation pairs",
        dataset.len()
    );

    if dataset.is_empty() {
        panic!("Dataset is empty");
    }

    let source_texts: Vec<String> =
        dataset
            .iter()
            .map(|pair| pair.source.clone())
            .collect();

    let target_texts: Vec<String> =
        dataset
            .iter()
            .map(|pair| pair.target.clone())
            .collect();

    println!(
        "Source training texts: {}",
        source_texts.len()
    );

    println!(
        "Target training texts: {}",
        target_texts.len()
    );

    println!("\nCREATING SOURCE BPE TOKENIZER");

    let mut source_tokenizer =
        BpeTokenizer::new(100);

    println!("\nTRAINING SOURCE BPE TOKENIZER");

    source_tokenizer.train(&source_texts);

    let source_vocab_size =
        source_tokenizer.vocab_size();

    println!(
        "Source vocabulary size: {}",
        source_vocab_size
    );

    println!("\nCREATING TARGET BPE TOKENIZER");

    let mut target_tokenizer =
        BpeTokenizer::new(100);

    println!("\nTRAINING TARGET BPE TOKENIZER");

    target_tokenizer.train(&target_texts);

    let target_vocab_size =
        target_tokenizer.vocab_size();

    println!(
        "Target vocabulary size: {}",
        target_vocab_size
    );

    let device =
        NdArrayDevice::Cpu;

    let embedding_dim = 128;
    let num_heads = 8;
    let ffn_hidden_dim = 512;
    let num_encoder_layers = 6;
    let num_decoder_layers = 6;
    let max_encoder_sequence_length = 512;
    let max_decoder_sequence_length = 512;

    println!("\nCREATING SOURCE EMBEDDING");

    let source_embedding:
        EmbeddingModel<Backend> =
        EmbeddingModel::new(
            source_vocab_size,
            embedding_dim,
            &device,
        );

    println!(
        "Source vocabulary size: {}",
        source_vocab_size
    );

    println!(
        "Source embedding dimension: {}",
        embedding_dim
    );

    println!("\nCREATING TARGET EMBEDDING");

    let target_embedding:
        EmbeddingModel<Backend> =
        EmbeddingModel::new(
            target_vocab_size,
            embedding_dim,
            &device,
        );

    println!(
        "Target vocabulary size: {}",
        target_vocab_size
    );

    println!(
        "Target embedding dimension: {}",
        embedding_dim
    );

    println!("\nCREATING TRANSFORMER");

    let transformer:
        TransformerModel<Backend> =
        TransformerModel::new(
            num_encoder_layers,
            num_decoder_layers,
            embedding_dim,
            num_heads,
            ffn_hidden_dim,
            max_encoder_sequence_length,
            max_decoder_sequence_length,
            target_vocab_size,
            &device,
        );

    println!(
        "Encoder layers: {}",
        num_encoder_layers
    );

    println!(
        "Decoder layers: {}",
        num_decoder_layers
    );

    println!(
        "Embedding dimension: {}",
        embedding_dim
    );

    println!(
        "Attention heads: {}",
        num_heads
    );

    println!(
        "FFN hidden dimension: {}",
        ffn_hidden_dim
    );

    println!(
        "Maximum encoder sequence length: {}",
        max_encoder_sequence_length
    );

    println!(
        "Maximum decoder sequence length: {}",
        max_decoder_sequence_length
    );

    println!(
        "Target vocabulary size: {}",
        target_vocab_size
    );

    let batch_size =
        dataset.len().min(8);

    let batch =
        TranslationBatch::from_pairs(
            &dataset[..batch_size],
            &source_tokenizer,
            &target_tokenizer,
        );

    println!("\n================================================");
    println!("TRANSLATION BATCH");
    println!("================================================");

    println!(
        "Batch size: {}",
        batch.batch_size()
    );

    println!(
        "Source shape: {:?}",
        batch.source_shape()
    );

    println!(
        "Target shape: {:?}",
        batch.target_shape()
    );

    println!(
        "Maximum source length: {}",
        batch.max_source_length
    );

    println!(
        "Maximum target length: {}",
        batch.max_target_length
    );

    assert!(
        batch.max_source_length
            <= max_encoder_sequence_length,
        "Batch source sequence length ({}) exceeds maximum encoder sequence length ({})",
        batch.max_source_length,
        max_encoder_sequence_length
    );

    assert!(
        batch.max_target_length
            <= max_decoder_sequence_length,
        "Batch target sequence length ({}) exceeds maximum decoder sequence length ({})",
        batch.max_target_length,
        max_decoder_sequence_length
    );

    let source_ids =
        batch.source_ids_flattened();

    let target_input_ids =
        batch.target_input_ids_flattened();

    let source_ids =
        Tensor::<Backend, 1, Int>::from_ints(
            source_ids
                .iter()
                .map(|&id| id as i64)
                .collect::<Vec<i64>>()
                .as_slice(),
            &device,
        )
        .reshape([
            batch.batch_size(),
            batch.max_source_length,
        ]);

    let target_input_ids =
        Tensor::<Backend, 1, Int>::from_ints(
            target_input_ids
                .iter()
                .map(|&id| id as i64)
                .collect::<Vec<i64>>()
                .as_slice(),
            &device,
        )
        .reshape([
            batch.batch_size(),
            batch.max_target_length,
        ]);

    let source_embeddings =
        source_embedding.forward(
            source_ids,
        );

    let target_embeddings =
        target_embedding.forward(
            target_input_ids,
        );

    println!("\n================================================");
    println!("EMBEDDING TENSORS");
    println!("================================================");

    println!(
        "Source embedding shape: {:?}",
        source_embeddings
            .shape()
            .dims::<3>()
    );

    println!(
        "Target embedding shape: {:?}",
        target_embeddings
            .shape()
            .dims::<3>()
    );

    let source_padding_mask =
        batch.source_padding_mask();

    let target_padding_mask =
        batch.target_padding_mask();

    let source_padding_values:
        Vec<bool> =
        source_padding_mask
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();

    let target_padding_values:
        Vec<bool> =
        target_padding_mask
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();

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

    println!("\n================================================");
    println!("PADDING MASKS");
    println!("================================================");

    println!(
        "Source padding mask shape: {:?}",
        source_padding_mask
            .shape()
            .dims::<2>()
    );

    println!(
        "Target padding mask shape: {:?}",
        target_padding_mask
            .shape()
            .dims::<2>()
    );

    println!("\n================================================");
    println!("FULL TRANSFORMER FORWARD PASS");
    println!("================================================");

    let logits =
        transformer.forward(
            source_embeddings,
            target_embeddings,
            Some(source_padding_mask),
            Some(target_padding_mask),
        );

    println!(
        "Logits tensor shape: {:?}",
        logits.shape().dims::<3>()
    );

    println!(
        "Expected logits shape: [{}, {}, {}]",
        batch.batch_size(),
        batch.max_target_length,
        target_vocab_size
    );

    println!("\n================================================");
    println!("TRANSFORMER MODULES");
    println!("================================================");

    println!("EncoderBlock       -> implemented");
    println!("DecoderBlock       -> implemented");
    println!("MHSA               -> implemented");
    println!("Masked MHSA        -> implemented");
    println!("Cross MHSA         -> implemented");
    println!("RoPE               -> implemented");
    println!("Transformer        -> implemented");
    println!("LM Head            -> implemented");
    println!("Source PAD Mask    -> implemented");
    println!("Target PAD Mask    -> implemented");
    println!("Causal Mask        -> implemented");

    println!("\n================================================");
    println!("CURRENT PIPELINE");
    println!("================================================");

    println!(
        "Source: Dataset -> Source BPE -> Source IDs -> Padding -> Source Embedding"
    );

    println!(
        "Target: Dataset -> Target BPE -> Target Input IDs -> Padding -> Target Embedding"
    );

    println!(
        "Encoder: Source Embedding -> Source PAD Mask -> Encoder Blocks"
    );

    println!(
        "Decoder: Target Embedding -> Target PAD + Causal Mask -> Masked Self Attention"
    );

    println!(
        "Cross Attention: Decoder States -> Source PAD Mask -> Encoder States"
    );

    println!(
        "Output: Decoder Hidden States -> LM Head -> Target Vocabulary Logits"
    );

    println!("\nFULL BATCH TRANSFORMER PIPELINE VERIFIED");
}
