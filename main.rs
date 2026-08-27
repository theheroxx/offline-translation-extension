mod FFN;
mod Tokenizer;

use Tokenizer::dataset::load_dataset;
use Tokenizer::tokenizer::Tokenizer as BpeTokenizer;

fn main() {
    // LOAD DATASET

    let dataset = load_dataset("data")
        .expect("Failed to load dataset");

    println!(
        "Loaded {} translation pairs",
        dataset.len()
    );

    // COLLECT SOURCE + TARGET TEXT

    let mut texts: Vec<String> = Vec::new();

    for pair in &dataset {
        texts.push(pair.source.clone());
        texts.push(pair.target.clone());
    }

    println!(
        "Total training texts: {}",
        texts.len()
    );

    // CREATE BPE TOKENIZER

    let mut tokenizer = BpeTokenizer::new(100);

    // TRAIN

    tokenizer.train(&texts);

    // PRINT VOCABULARY

    tokenizer.print_vocab();

    // PRINT MERGES

    tokenizer.print_merges();

    // TEST

    if let Some(pair) = dataset.first() {
        println!();
        println!("========================================");
        println!("TOKENIZER TEST");
        println!("========================================");

        println!();
        println!("SOURCE:");
        println!("{}", pair.source);

        // TOKENIZE

        let tokens = tokenizer.tokenize(&pair.source);

        println!();
        println!("TOKENS:");
        println!("{:?}", tokens);

        // ENCODE

        let encoded = tokenizer.encode(&pair.source);

        println!();
        println!("ENCODED:");
        println!("{:?}", encoded);

        // DECODE

        let decoded = tokenizer.decode(&encoded);

        println!();
        println!("DECODED:");
        println!("{}", decoded);

        // TOKEN DEBUG

        tokenizer.print_tokens(&pair.source);
    }
}