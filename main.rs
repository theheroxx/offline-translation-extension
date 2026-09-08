mod FFN;
mod Tokenizer;
mod Embedding;

use Tokenizer::dataset::load_dataset;
use Tokenizer::tokenizer::Tokenizer as BpeTokenizer;
use Embedding::Embedding as EmbeddingModel;
use burn_ndarray::NdArrayDevice;

fn main() {
    // Load dataset
    println!("LOADING DATASET");

    let dataset = load_dataset("data")
        .expect("Failed to load dataset");

    println!(
        "Loaded {} translation pairs",
        dataset.len()
    );

    // Collect source and target texts

    let mut texts: Vec<String> = Vec::new();

    for pair in &dataset {
        texts.push(pair.source.clone());
        texts.push(pair.target.clone());
    }

    println!(
        "Total training texts: {}",
        texts.len()
    );

    // Create BPE tokenizer
    println!("\nCREATING BPE TOKENIZER");

    let mut tokenizer = BpeTokenizer::new(100);

    // Train BPE tokenizer
    println!("\nTRAINING BPE TOKENIZER");

    tokenizer.train(&texts);

    println!();
    println!(
        "Final vocabulary size: {}",
        tokenizer.vocab_size()
    );

    // Create embedding model
    println!("\nCREATING EMBEDDING MODEL");

    let vocab_size = tokenizer.vocab_size();
    let embedding_dim = 128;

    let device = NdArrayDevice::Cpu;

    let embedding: EmbeddingModel<burn_ndarray::NdArray<f32>> = EmbeddingModel::new(
        vocab_size,
        embedding_dim,
        &device,
    );

    println!(
        "Vocabulary size: {}",
        vocab_size
    );

    println!(
        "Embedding dimension: {}",
        embedding_dim
    );

    // Tokenizer + embedding test
    if let Some(pair) = dataset.first() {
        println!("\nTOKENIZER + EMBEDDING TEST");

        // Source
        println!("\nSOURCE:");
        println!("{}", pair.source);

        // Tokenize
        let tokens = tokenizer.tokenize(&pair.source);

        println!();
        println!("TOKENS:");
        println!("{:?}", tokens);

        // Encode
        let token_ids = tokenizer.encode(&pair.source);

        println!();
        println!("TOKEN IDS:");
        println!("{:?}", token_ids);

        // Decode
        let decoded = tokenizer.decode(&token_ids);

        println!();
        println!("DECODED:");
        println!("{}", decoded);

        // Embedding
        println!("\nEMBEDDING:");

        let embeddings = embedding.forward_sequence(&token_ids, &device);

        println!("Input token count: {}", token_ids.len());
        println!("Embedding output count: {}", embeddings.shape().dims[0]);
        println!("Embedding output is a tensor; inspect via tensor APIs if needed");
    }

    // Transformer status
    println!("\nTRANSFORMER MODULES");
    println!("MHSA -> implemented");
    println!("Masked MHSA -> implemented");
    println!("Cross MHSA -> implemented");
    println!("RoPE -> next integration step");

    // Pipeline
    println!("\nCURRENT PIPELINE: Dataset -> BPE Tokenizer -> Token IDs -> Embedding -> Transformer -> Output");
}
