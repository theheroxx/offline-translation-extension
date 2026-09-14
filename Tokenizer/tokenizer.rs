use std::collections::HashMap;

const END_OF_WORD: &str = "</w>";
const PAD_TOKEN: &str = "<PAD>";
const UNK_TOKEN: &str = "<UNK>";
const BOS_TOKEN: &str = "<BOS>";
const EOS_TOKEN: &str = "<EOS>";

#[derive(Debug, Clone)]
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
        println!("========================================");
        println!("BPE TRAINING");
        println!("========================================");
        println!("Training texts: {}", texts.len());
        println!("Requested merges: {}", self.num_merges);

        self.vocab.clear();
        self.id_to_token.clear();
        self.merges.clear();

        self.add_special_tokens();

        let mut word_frequencies: HashMap<Vec<String>, usize> =
            HashMap::new();

        for text in texts {
            for word in text.split_whitespace() {
                if word.is_empty() {
                    continue;
                }

                let symbols = Self::word_to_symbols(word);

                *word_frequencies
                    .entry(symbols)
                    .or_insert(0) += 1;
            }
        }

        println!(
            "Initial unique word representations: {}",
            word_frequencies.len()
        );

        if word_frequencies.is_empty() {
            println!("No training data found.");
            return;
        }

        let mut initial_symbols: Vec<String> = Vec::new();

        for symbols in word_frequencies.keys() {
            for symbol in symbols {
                if !initial_symbols.contains(symbol) {
                    initial_symbols.push(symbol.clone());
                }
            }
        }

        initial_symbols.sort();

        for symbol in &initial_symbols {
            self.add_token(symbol);
        }

        for merge_number in 0..self.num_merges {
            let pair_counts = Self::count_pairs(&word_frequencies);

            if pair_counts.is_empty() {
                println!("No more pairs available.");
                break;
            }

            let best_pair = match pair_counts
                .iter()
                .max_by(|a, b| {
                    a.1.cmp(b.1)
                        .then_with(|| b.0.cmp(a.0))
                })
            {
                Some((pair, _)) => pair.clone(),
                None => break,
            };

            let count =
                pair_counts.get(&best_pair).copied().unwrap_or(0);

            println!(
                "Merge {:>4}: {:?} + {:?} -> count {}",
                merge_number + 1,
                best_pair.0,
                best_pair.1,
                count
            );

            self.merges.push((
                best_pair.0.clone(),
                best_pair.1.clone(),
            ));

            let merged_token = format!(
                "{}{}",
                best_pair.0,
                best_pair.1
            );

            self.add_token(&merged_token);

            let mut new_word_frequencies:
                HashMap<Vec<String>, usize> =
                HashMap::new();

            for (symbols, frequency) in &word_frequencies {
                let merged = Self::merge_pair(
                    symbols,
                    &best_pair.0,
                    &best_pair.1,
                );

                *new_word_frequencies
                    .entry(merged)
                    .or_insert(0) += *frequency;
            }

            word_frequencies = new_word_frequencies;
        }

        println!();
        println!("BPE training complete.");
        println!("Vocabulary size: {}", self.vocab.len());
        println!("Number of merges: {}", self.merges.len());
    }

    fn word_to_symbols(word: &str) -> Vec<String> {
        let chars: Vec<char> = word.chars().collect();

        if chars.is_empty() {
            return Vec::new();
        }

        let mut symbols =
            Vec::with_capacity(chars.len());

        for (i, ch) in chars.iter().enumerate() {
            if i == chars.len() - 1 {
                symbols.push(
                    format!("{}{}", ch, END_OF_WORD)
                );
            } else {
                symbols.push(ch.to_string());
            }
        }

        symbols
    }

    fn count_pairs(
        word_frequencies: &HashMap<Vec<String>, usize>,
    ) -> HashMap<(String, String), usize> {
        let mut pair_counts:
            HashMap<(String, String), usize> =
            HashMap::new();

        for (symbols, frequency) in word_frequencies {
            if symbols.len() < 2 {
                continue;
            }

            for i in 0..symbols.len() - 1 {
                let pair = (
                    symbols[i].clone(),
                    symbols[i + 1].clone(),
                );

                *pair_counts
                    .entry(pair)
                    .or_insert(0) += *frequency;
            }
        }

        pair_counts
    }

    fn merge_pair(
        symbols: &[String],
        first: &str,
        second: &str,
    ) -> Vec<String> {
        if symbols.len() < 2 {
            return symbols.to_vec();
        }

        let mut result =
            Vec::with_capacity(symbols.len());

        let mut i = 0;

        while i < symbols.len() {
            if i + 1 < symbols.len()
                && symbols[i] == first
                && symbols[i + 1] == second
            {
                result.push(format!(
                    "{}{}",
                    symbols[i],
                    symbols[i + 1]
                ));

                i += 2;
            } else {
                result.push(symbols[i].clone());
                i += 1;
            }
        }

        result
    }

    fn apply_merges(
        &self,
        word: &str,
    ) -> Vec<String> {
        let mut symbols =
            Self::word_to_symbols(word);

        for (first, second) in &self.merges {
            symbols = Self::merge_pair(
                &symbols,
                first,
                second,
            );
        }

        symbols
    }

    fn encode_content(
        &self,
        text: &str,
    ) -> Vec<usize> {
        let mut ids = Vec::new();

        let unk_id = self.unk_id();

        for word in text.split_whitespace() {
            if word.is_empty() {
                continue;
            }

            let symbols =
                self.apply_merges(word);

            for symbol in symbols {
                let id =
                    self.vocab
                        .get(&symbol)
                        .copied()
                        .unwrap_or(unk_id);

                ids.push(id);
            }
        }

        ids
    }

    pub fn encode(
        &self,
        text: &str,
    ) -> Vec<usize> {
        let mut ids =
            Vec::new();

        ids.push(self.bos_id());

        ids.extend(
            self.encode_content(text)
        );

        ids.push(self.eos_id());

        ids
    }

    pub fn encode_source(
        &self,
        text: &str,
    ) -> Vec<usize> {
        let mut ids =
            self.encode_content(text);

        ids.push(self.eos_id());

        ids
    }

    pub fn encode_target_input(
        &self,
        text: &str,
    ) -> Vec<usize> {
        let mut ids =
            Vec::new();

        ids.push(self.bos_id());

        ids.extend(
            self.encode_content(text)
        );

        ids
    }

    pub fn encode_target_labels(
        &self,
        text: &str,
    ) -> Vec<usize> {
        let mut ids =
            self.encode_content(text);

        ids.push(self.eos_id());

        ids
    }

    pub fn decode(
        &self,
        ids: &[usize],
    ) -> String {
        let mut output =
            String::new();

        for &id in ids {
            if id >= self.id_to_token.len() {
                continue;
            }

            let token =
                &self.id_to_token[id];

            match token.as_str() {
                PAD_TOKEN |
                BOS_TOKEN |
                EOS_TOKEN => {}

                UNK_TOKEN => {
                    output.push('�');
                }

                token => {
                    output.push_str(token);
                }
            }
        }

        output
            .replace(END_OF_WORD, " ")
            .trim()
            .to_string()
    }

    pub fn tokenize(
        &self,
        text: &str,
    ) -> Vec<String> {
        let mut tokens =
            Vec::new();

        for word in text.split_whitespace() {
            if word.is_empty() {
                continue;
            }

            let symbols =
                self.apply_merges(word);

            tokens.extend(symbols);
        }

        tokens
    }

    pub fn token_to_id(
        &self,
        token: &str,
    ) -> Option<usize> {
        self.vocab
            .get(token)
            .copied()
    }

    pub fn id_to_token(
        &self,
        id: usize,
    ) -> Option<&str> {
        self.id_to_token
            .get(id)
            .map(|token| token.as_str())
    }

    pub fn pad_id(&self) -> usize {
        self.token_to_id(PAD_TOKEN)
            .expect("PAD token is missing")
    }

    pub fn unk_id(&self) -> usize {
        self.token_to_id(UNK_TOKEN)
            .expect("UNK token is missing")
    }

    pub fn bos_id(&self) -> usize {
        self.token_to_id(BOS_TOKEN)
            .expect("BOS token is missing")
    }

    pub fn eos_id(&self) -> usize {
        self.token_to_id(EOS_TOKEN)
            .expect("EOS token is missing")
    }

    pub fn special_token_ids(
        &self,
    ) -> (usize, usize, usize, usize) {
        (
            self.pad_id(),
            self.unk_id(),
            self.bos_id(),
            self.eos_id(),
        )
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    pub fn print_vocab(&self) {
        println!();
        println!("========================================");
        println!("VOCABULARY");
        println!("========================================");

        for (id, token)
            in self.id_to_token.iter().enumerate()
        {
            println!(
                "{:>5} -> {:?}",
                id,
                token
            );
        }
    }

    pub fn print_merges(&self) {
        println!();
        println!("========================================");
        println!("BPE MERGES");
        println!("========================================");

        for (i, (first, second))
            in self.merges.iter().enumerate()
        {
            println!(
                "{:>5}: {:?} + {:?}",
                i,
                first,
                second
            );
        }
    }

    pub fn print_tokens(
        &self,
        text: &str,
    ) {
        println!();
        println!("========================================");
        println!("TOKENIZATION");
        println!("========================================");

        println!("Input:");
        println!("{}", text);

        println!();

        let tokens =
            self.tokenize(text);

        for (position, token)
            in tokens.iter().enumerate()
        {
            let id =
                self.token_to_id(token);

            println!(
                "{:>5}: {:?} -> ID {:?}",
                position,
                token,
                id
            );
        }

        println!();
        println!(
            "Total tokens: {}",
            tokens.len()
        );
    }
}
