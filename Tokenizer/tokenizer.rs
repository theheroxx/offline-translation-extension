use std::collections::HashMap;

const END_OF_WORD: &str = "</w>";

#[derive(Debug, Clone)]
pub struct Tokenizer {
    pub vocab: HashMap<String, usize>,
    pub id_to_token: Vec<String>,
    pub merges: Vec<(String, String)>,
    pub num_merges: usize,
}

impl Tokenizer {

    // CREATE TOKENIZER
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

    // SPECIAL TOKENS
    fn add_special_tokens(&mut self) {
        self.add_token("<PAD>");
        self.add_token("<UNK>");
        self.add_token("<BOS>");
        self.add_token("<EOS>");
    }

    // ADD TOKEN
    fn add_token(&mut self, token: &str) -> usize {
        if let Some(&id) = self.vocab.get(token) {
            return id;
        }

        let id = self.id_to_token.len();

        self.vocab.insert(token.to_string(), id);
        self.id_to_token.push(token.to_string());

        id
    }

    // TRAIN BPE
    pub fn train(&mut self, texts: &[String]) {
        println!("========================================");
        println!("BPE TRAINING");
        println!("========================================");
        println!("Training texts: {}", texts.len());
        println!("Requested merges: {}", self.num_merges);

        // Reset tokenizer
        self.vocab.clear();
        self.id_to_token.clear();
        self.merges.clear();

        self.add_special_tokens();

        // WORD FREQUENCIES
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

        // INITIAL CHARACTER VOCABULARY
        let mut initial_symbols: Vec<String> = Vec::new();

        for symbols in word_frequencies.keys() {
            for symbol in symbols {
                if !initial_symbols.contains(symbol) {
                    initial_symbols.push(symbol.clone());
                }
            }
        }

        // Sort for deterministic IDs
        initial_symbols.sort();

        for symbol in &initial_symbols {
            self.add_token(symbol);
        }

        // BPE MERGE LOOP

        for merge_number in 0..self.num_merges {
            let pair_counts =
                Self::count_pairs(&word_frequencies);

            if pair_counts.is_empty() {
                println!("No more pairs available.");
                break;
            }

            // Deterministic best-pair selection.
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

            // Save merge
            self.merges.push((
                best_pair.0.clone(),
                best_pair.1.clone(),
            ));

            // Add resulting merged token to vocabulary
            let merged_token = format!(
                "{}{}",
                best_pair.0,
                best_pair.1
            );

            self.add_token(&merged_token);

            // Apply merge to all words

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

    // WORD -> INITIAL SYMBOLS

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

    // COUNT ADJACENT PAIRS

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

    // MERGE ONE PAIR

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

    // APPLY LEARNED MERGES

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

    // ENCODE

    pub fn encode(
        &self,
        text: &str,
    ) -> Vec<usize> {
        let mut ids = Vec::new();

        // BOS
        if let Some(&bos_id) =
            self.vocab.get("<BOS>")
        {
            ids.push(bos_id);
        }

        let unk_id =
            match self.vocab.get("<UNK>") {
                Some(&id) => id,
                None => return ids,
            };

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

        // EOS
        if let Some(&eos_id) =
            self.vocab.get("<EOS>")
        {
            ids.push(eos_id);
        }

        ids
    }

    // DECODE

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
                "<PAD>" |
                "<BOS>" |
                "<EOS>" => {}

                "<UNK>" => {
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

    // TOKENIZE

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

    // TOKEN -> ID

    pub fn token_to_id(
        &self,
        token: &str,
    ) -> Option<usize> {
        self.vocab
            .get(token)
            .copied()
    }

    // ID -> TOKEN

    pub fn id_to_token(
        &self,
        id: usize,
    ) -> Option<&str> {
        self.id_to_token
            .get(id)
            .map(|token| token.as_str())
    }

    // VOCABULARY SIZE

    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    // PRINT VOCABULARY

    pub fn print_vocab(&self) {
        println!();
        println!("========================================");
        println!("VOCABULARY");
        println!("========================================");

        for (id, token) in
            self.id_to_token.iter().enumerate()
        {
            println!(
                "{:>5} -> {:?}",
                id,
                token
            );
        }
    }

    // PRINT MERGES

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

    // PRINT TOKENS

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
        println!("Total tokens: {}", tokens.len());
    }
}