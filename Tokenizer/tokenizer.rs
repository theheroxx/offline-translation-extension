use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

const END_OF_WORD: &str = "</w>";
const PAD_TOKEN: &str = "<PAD>";
const UNK_TOKEN: &str = "<UNK>";
const BOS_TOKEN: &str = "<BOS>";
const EOS_TOKEN: &str = "<EOS>";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tokenizer {
    pub vocab: HashMap<String, usize>,
    pub id_to_token: Vec<String>,
    pub merges: Vec<(String, String)>,
    pub num_merges: usize,
}

impl Tokenizer {
    pub fn new(num_merges: usize) -> Self {
        let mut tokenizer = Self {
            vocab: HashMap::new(),
            id_to_token: Vec::new(),
            merges: Vec::new(),
            num_merges,
        };

        tokenizer.add_special_tokens();
        tokenizer
    }

    fn add_special_tokens(&mut self) {
        self.add_token(PAD_TOKEN);
        self.add_token(UNK_TOKEN);
        self.add_token(BOS_TOKEN);
        self.add_token(EOS_TOKEN);
    }

    fn add_token(&mut self, token: &str) -> usize {
        if let Some(&id) = self.vocab.get(token) {
            return id;
        }

        let id = self.id_to_token.len();

        self.vocab.insert(token.to_string(), id);
        self.id_to_token.push(token.to_string());

        id
    }

    pub fn train(&mut self, texts: &[String]) {
        self.vocab.clear();
        self.id_to_token.clear();
        self.merges.clear();

        self.add_special_tokens();

        let mut word_frequencies: HashMap<Vec<String>, usize> = HashMap::new();

        for text in texts {
            for word in text.split_whitespace() {
                if word.is_empty() {
                    continue;
                }

                let symbols = Self::word_to_symbols(word);

                *word_frequencies.entry(symbols).or_insert(0) += 1;
            }
        }

        if word_frequencies.is_empty() {
            return;
        }

        let mut initial_symbols = Vec::new();

        for symbols in word_frequencies.keys() {
            for symbol in symbols {
                initial_symbols.push(symbol.clone());
            }
        }

        initial_symbols.sort();
        initial_symbols.dedup();

        for symbol in &initial_symbols {
            self.add_token(symbol);
        }

        for _ in 0..self.num_merges {
            let pair_counts = Self::count_pairs(&word_frequencies);

            if pair_counts.is_empty() {
                break;
            }

            let best_pair = match pair_counts
                .iter()
                .max_by(|a, b| a.1.cmp(b.1).then_with(|| b.0.cmp(a.0)))
            {
                Some((pair, _)) => pair.clone(),
                None => break,
            };

            self.merges.push(best_pair.clone());

            let merged_token = format!("{}{}", best_pair.0, best_pair.1);

            self.add_token(&merged_token);

            let mut new_word_frequencies: HashMap<Vec<String>, usize> = HashMap::new();

            for (symbols, frequency) in &word_frequencies {
                let merged = Self::merge_pair(symbols, &best_pair.0, &best_pair.1);

                *new_word_frequencies.entry(merged).or_insert(0) += *frequency;
            }

            word_frequencies = new_word_frequencies;
        }
    }

    fn word_to_symbols(word: &str) -> Vec<String> {
        let chars: Vec<char> = word.chars().collect();

        if chars.is_empty() {
            return Vec::new();
        }

        let mut symbols = Vec::with_capacity(chars.len());

        for (index, character) in chars.iter().enumerate() {
            if index == chars.len() - 1 {
                symbols.push(format!("{}{}", character, END_OF_WORD));
            } else {
                symbols.push(character.to_string());
            }
        }

        symbols
    }

    fn count_pairs(
        word_frequencies: &HashMap<Vec<String>, usize>,
    ) -> HashMap<(String, String), usize> {
        let mut pair_counts: HashMap<(String, String), usize> = HashMap::new();

        for (symbols, frequency) in word_frequencies {
            if symbols.len() < 2 {
                continue;
            }

            for index in 0..symbols.len() - 1 {
                let pair = (symbols[index].clone(), symbols[index + 1].clone());

                *pair_counts.entry(pair).or_insert(0) += *frequency;
            }
        }

        pair_counts
    }

    fn merge_pair(symbols: &[String], first: &str, second: &str) -> Vec<String> {
        if symbols.len() < 2 {
            return symbols.to_vec();
        }

        let mut result = Vec::with_capacity(symbols.len());
        let mut index = 0;

        while index < symbols.len() {
            if index + 1 < symbols.len() && symbols[index] == first && symbols[index + 1] == second
            {
                result.push(format!("{}{}", symbols[index], symbols[index + 1]));

                index += 2;
            } else {
                result.push(symbols[index].clone());
                index += 1;
            }
        }

        result
    }

    fn apply_merges(&self, word: &str) -> Vec<String> {
        let mut symbols = Self::word_to_symbols(word);

        for (first, second) in &self.merges {
            symbols = Self::merge_pair(&symbols, first, second);
        }

        symbols
    }

    fn encode_content(&self, text: &str) -> Vec<usize> {
        let mut ids = Vec::new();
        let unk_id = self.unk_id();

        for word in text.split_whitespace() {
            if word.is_empty() {
                continue;
            }

            let symbols = self.apply_merges(word);

            for symbol in symbols {
                let id = self.vocab.get(&symbol).copied().unwrap_or(unk_id);

                ids.push(id);
            }
        }

        ids
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut ids = Vec::new();

        ids.push(self.bos_id());
        ids.extend(self.encode_content(text));
        ids.push(self.eos_id());

        ids
    }

    pub fn encode_source(&self, text: &str) -> Vec<usize> {
        let mut ids = self.encode_content(text);
        ids.push(self.eos_id());

        ids
    }

    pub fn encode_target_input(&self, text: &str) -> Vec<usize> {
        let mut ids = Vec::new();

        ids.push(self.bos_id());
        ids.extend(self.encode_content(text));

        ids
    }

    pub fn encode_target_labels(&self, text: &str) -> Vec<usize> {
        let mut ids = self.encode_content(text);
        ids.push(self.eos_id());

        ids
    }

    pub fn count_source_tokens(&self, text: &str) -> usize {
        self.encode_source(text).len()
    }

    pub fn count_target_tokens(&self, text: &str) -> usize {
        self.encode_target_input(text).len()
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        let mut output = String::new();

        for &id in ids {
            let Some(token) = self.id_to_token.get(id) else {
                continue;
            };

            match token.as_str() {
                PAD_TOKEN | BOS_TOKEN | EOS_TOKEN => {}
                UNK_TOKEN => output.push('�'),
                token => output.push_str(token),
            }
        }

        output
            .replace(END_OF_WORD, " ")
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    pub fn tokenize(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();

        for word in text.split_whitespace() {
            if word.is_empty() {
                continue;
            }

            tokens.extend(self.apply_merges(word));
        }

        tokens
    }

    pub fn token_to_id(&self, token: &str) -> Option<usize> {
        self.vocab.get(token).copied()
    }

    pub fn id_to_token(&self, id: usize) -> Option<&str> {
        self.id_to_token.get(id).map(|token| token.as_str())
    }

    pub fn pad_id(&self) -> usize {
        self.token_to_id(PAD_TOKEN).expect("PAD token is missing")
    }

    pub fn unk_id(&self) -> usize {
        self.token_to_id(UNK_TOKEN).expect("UNK token is missing")
    }

    pub fn bos_id(&self) -> usize {
        self.token_to_id(BOS_TOKEN).expect("BOS token is missing")
    }

    pub fn eos_id(&self) -> usize {
        self.token_to_id(EOS_TOKEN).expect("EOS token is missing")
    }

    pub fn special_token_ids(&self) -> (usize, usize, usize, usize) {
        (self.pad_id(), self.unk_id(), self.bos_id(), self.eos_id())
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.vocab.len() != self.id_to_token.len() {
            return Err("Vocabulary and ID-to-token mapping have different lengths".to_string());
        }

        for (id, token) in self.id_to_token.iter().enumerate() {
            match self.vocab.get(token) {
                Some(&mapped_id) if mapped_id == id => {}
                _ => {
                    return Err(format!(
                        "Invalid vocabulary mapping for token {:?} at ID {}",
                        token, id
                    ));
                }
            }
        }

        if self.token_to_id(PAD_TOKEN).is_none()
            || self.token_to_id(UNK_TOKEN).is_none()
            || self.token_to_id(BOS_TOKEN).is_none()
            || self.token_to_id(EOS_TOKEN).is_none()
        {
            return Err("One or more special tokens are missing".to_string());
        }

        for (index, (first, second)) in self.merges.iter().enumerate() {
            let merged_token = format!("{}{}", first, second);

            if !self.vocab.contains_key(&merged_token) {
                return Err(format!(
                    "Merge {} produces missing vocabulary token {:?}",
                    index, merged_token
                ));
            }
        }

        Ok(())
    }

    pub fn save_json<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn Error>> {
        self.validate().map_err(std::io::Error::other)?;

        let file = File::create(path)?;
        let writer = BufWriter::new(file);

        serde_json::to_writer_pretty(writer, self)?;

        Ok(())
    }

    pub fn load_json<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let tokenizer: Self = serde_json::from_reader(reader)?;

        tokenizer.validate().map_err(std::io::Error::other)?;

        Ok(tokenizer)
    }

    pub fn print_vocab(&self) {
        println!("========================================");
        println!("VOCABULARY");
        println!("========================================");

        for (id, token) in self.id_to_token.iter().enumerate() {
            println!("{:>5} -> {:?}", id, token);
        }
    }

    pub fn print_merges(&self) {
        println!("========================================");
        println!("BPE MERGES");
        println!("========================================");

        for (index, (first, second)) in self.merges.iter().enumerate() {
            println!("{:>5}: {:?} + {:?}", index, first, second);
        }
    }

    pub fn print_tokens(&self, text: &str) {
        println!("========================================");
        println!("TOKENIZATION");
        println!("========================================");

        println!("Input:");
        println!("{}", text);
        println!();

        let tokens = self.tokenize(text);

        for (position, token) in tokens.iter().enumerate() {
            let id = self.token_to_id(token);

            println!("{:>5}: {:?} -> ID {:?}", position, token, id);
        }

        println!();
        println!("Total tokens: {}", tokens.len());
    }
}
