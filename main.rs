mod FFN;
mod Tokenizer;
mod Embedding;
mod Attention;
mod PositionalEncoding;
mod Transformer;

use Tokenizer::dataset::load_dataset;
use Tokenizer::tokenizer::Tokenizer as BpeTokenizer;
use Embedding::Embedding as EmbeddingModel;

use burn_ndarray::NdArrayDevice;

type Backend = burn_ndarray::NdArray<f32>;

fn main() {
println!("LOADING DATASET");

let dataset = load_dataset("data")
    .expect("Failed to load dataset");

println!("Loaded {} translation pairs", dataset.len());

let mut source_texts: Vec<String> = Vec::new();
let mut target_texts: Vec<String> = Vec::new();

for pair in &dataset {
    source_texts.push(pair.source.clone());
    target_texts.push(pair.target.clone());
}

println!("Source training texts: {}", source_texts.len());
println!("Target training texts: {}", target_texts.len());

println!("\nCREATING SOURCE BPE TOKENIZER");

let mut source_tokenizer = BpeTokenizer::new(100);

println!("\nTRAINING SOURCE BPE TOKENIZER");

source_tokenizer.train(&source_texts);

let source_vocab_size = source_tokenizer.vocab_size();

println!(
    "Source vocabulary size: {}",
    source_vocab_size
);

println!("\nCREATING TARGET BPE TOKENIZER");

let mut target_tokenizer = BpeTokenizer::new(100);

println!("\nTRAINING TARGET BPE TOKENIZER");

target_tokenizer.train(&target_texts);

let target_vocab_size = target_tokenizer.vocab_size();

println!(
    "Target vocabulary size: {}",
    target_vocab_size
);

let device = NdArrayDevice::Cpu;

let embedding_dim = 128;

println!("\nCREATING SOURCE EMBEDDING");

let source_embedding: EmbeddingModel<Backend> =
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

let target_embedding: EmbeddingModel<Backend> =
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

if let Some(pair) = dataset.first() {
    println!("\n================================================");
    println!("SOURCE TOKENIZER + EMBEDDING TEST");
    println!("================================================");

    println!("\nSOURCE:");
    println!("{}", pair.source);

    let source_tokens =
        source_tokenizer.tokenize(&pair.source);

    println!("\nSOURCE TOKENS:");
    println!("{:?}", source_tokens);

    let source_token_ids =
        source_tokenizer.encode(&pair.source);

    println!("\nSOURCE TOKEN IDS:");
    println!("{:?}", source_token_ids);

    let source_decoded =
        source_tokenizer.decode(&source_token_ids);

    println!("\nSOURCE DECODED:");
    println!("{}", source_decoded);

    let source_embeddings =
        source_embedding.forward_sequence(
            &source_token_ids,
            &device,
        );

    println!("\nSOURCE EMBEDDING:");

    println!(
        "Input token count: {}",
        source_token_ids.len()
    );

    println!(
        "Embedding tensor shape: {:?}",
        source_embeddings.shape().dims::<3>()
    );
}

if let Some(pair) = dataset.first() {
    println!("\n================================================");
    println!("TARGET TOKENIZER + EMBEDDING TEST");
    println!("================================================");

    println!("\nTARGET:");
    println!("{}", pair.target);

    let target_tokens =
        target_tokenizer.tokenize(&pair.target);

    println!("\nTARGET TOKENS:");
    println!("{:?}", target_tokens);

    let target_token_ids =
        target_tokenizer.encode(&pair.target);

    println!("\nTARGET TOKEN IDS:");
    println!("{:?}", target_token_ids);

    let target_decoded =
        target_tokenizer.decode(&target_token_ids);

    println!("\nTARGET DECODED:");
    println!("{}", target_decoded);

    let target_embeddings =
        target_embedding.forward_sequence(
            &target_token_ids,
            &device,
        );

    println!("\nTARGET EMBEDDING:");

    println!(
        "Input token count: {}",
        target_token_ids.len()
    );

    println!(
        "Embedding tensor shape: {:?}",
        target_embeddings.shape().dims::<3>()
    );
}

println!("\n================================================");
println!("TRANSFORMER MODULES");
println!("================================================");

println!("EncoderBlock       -> implemented");
println!("DecoderBlock       -> implemented");
println!("MHSA               -> implemented");
println!("Masked MHSA        -> implemented");
println!("Cross MHSA         -> implemented");
println!("RoPE               -> next integration step");
println!("Transformer        -> not yet implemented");
println!("LM Head            -> not yet implemented");

println!("\n================================================");
println!("CURRENT PIPELINE");
println!("================================================");

println!(
    "Source: Dataset -> Source BPE -> Source IDs -> Source Embedding"
);

println!(
    "Target: Dataset -> Target BPE -> Target IDs -> Target Embedding"
);

println!(
    "Then: Source Embedding -> Encoder -> Decoder -> LM Head"
);

}
