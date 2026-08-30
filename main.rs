mod FFN;
mod Tokenizer;
mod Embedding;

use Tokenizer::dataset::load_dataset;
use Tokenizer::tokenizer::Tokenizer as BpeTokenizer;
use Embedding::Embedding as EmbeddingModel;

fn main() {

    // 1. LOAD DATASET

    println!("LOADING DATASET");

    let dataset = load_dataset("data")
        .expect("Failed to load dataset");

    println!(
        "Loaded {} translation pairs",
        dataset.len()
    );

    // 2. COLLECT SOURCE + TARGET TEXT

    let mut texts: Vec<String> = Vec::new();

    for pair in &dataset {
        texts.push(pair.source.clone());
        texts.push(pair.target.clone());
    }

    println!(
        "Total training texts: {}",
        texts.len()
    );

    // 3. CREATE BPE TOKENIZER

    println!();
    println!("CREATING BPE TOKENIZER");

    let mut tokenizer = BpeTokenizer::new(100);

    // 4. TRAIN BPE

    tokenizer.train(&texts);

    println!();
    println!(
        "Final vocabulary size: {}",
        tokenizer.vocab_size()
    );

    // 5. CREATE EMBEDDING MODEL

    println!();
    println!("CREATING EMBEDDING MODEL");

    let vocab_size = tokenizer.vocab_size();
    let embedding_dim = 128;

    let embedding = EmbeddingModel::new(
        vocab_size,
        embedding_dim,
    );

    println!(
        "Vocabulary size: {}",
        vocab_size
    );

    println!(
        "Embedding dimension: {}",
        embedding_dim
    );

    // 6. TEST TOKENIZER + EMBEDDING

    if let Some(pair) = dataset.first() {
        println!();
        println!("TOKENIZER + EMBEDDING TEST");

        println!();
        println!("SOURCE:");
        println!("{}", pair.source);

        // TOKENIZE

        let tokens = tokenizer.tokenize(&pair.source);

        println!();
        println!("TOKENS:");
        println!("{:?}", tokens);

        // ENCODE

        let token_ids = tokenizer.encode(&pair.source);

        println!();
        println!("TOKEN IDS:");
        println!("{:?}", token_ids);

        // DECODE

        let decoded = tokenizer.decode(&token_ids);

        println!();
        println!("DECODED:");
        println!("{}", decoded);

        // EMBEDDING

        println!();
        println!("EMBEDDING");

    let embeddings = embedding.forward(&token_ids);

    println!(
        "Input token count: {}",
        token_ids.len()
    );

    println!(
        "Embedding output count: {}",
        embeddings.len()
    );

    for (i, (token_id, vector)) in token_ids
        .iter()
        .zip(embeddings.iter())
        .enumerate()
    {
        println!();
        println!(
            "Token {} | ID {} | Embedding dimension {}",
            i,
            token_id,
            vector.len()
        );

        println!(
            "First values: {:?}",
            &vector[..vector.len().min(10)]
        );
    }
    }
}