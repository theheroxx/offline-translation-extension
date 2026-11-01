use crate::Tokenizer::dataset::TranslationPair;
use crate::Tokenizer::tokenizer::Tokenizer;

#[derive(Debug, Clone)]
pub struct TranslationSegment {
    pub source: String,
    pub target: String,
}

pub trait TokenCounter {
    fn count(&self, text: &str) -> usize;
}

#[derive(Clone, Copy)]
pub enum TokenSequenceType {
    Source,
    Target,
}

pub struct BpeTokenCounter<'a> {
    tokenizer: &'a Tokenizer,
    sequence_type: TokenSequenceType,
}

impl<'a> BpeTokenCounter<'a> {
    pub fn source(tokenizer: &'a Tokenizer) -> Self {
        Self {
            tokenizer,
            sequence_type: TokenSequenceType::Source,
        }
    }

    pub fn target(tokenizer: &'a Tokenizer) -> Self {
        Self {
            tokenizer,
            sequence_type: TokenSequenceType::Target,
        }
    }
}

impl TokenCounter for BpeTokenCounter<'_> {
    fn count(&self, text: &str) -> usize {
        match self.sequence_type {
            TokenSequenceType::Source => self.tokenizer.count_source_tokens(text),
            TokenSequenceType::Target => self.tokenizer.count_target_tokens(text),
        }
    }
}

pub struct WhitespaceTokenCounter;

impl TokenCounter for WhitespaceTokenCounter {
    fn count(&self, text: &str) -> usize {
        text.split_whitespace().count()
    }
}

pub struct DocumentSegmenter<S, T>
where
    S: TokenCounter,
    T: TokenCounter,
{
    max_source_tokens: usize,
    max_target_tokens: usize,
    source_token_counter: S,
    target_token_counter: T,
}

impl<S, T> DocumentSegmenter<S, T>
where
    S: TokenCounter,
    T: TokenCounter,
{
    pub fn new(
        max_source_tokens: usize,
        max_target_tokens: usize,
        source_token_counter: S,
        target_token_counter: T,
    ) -> Self {
        assert!(
            max_source_tokens > 0,
            "Maximum source tokens must be greater than zero"
        );

        assert!(
            max_target_tokens > 0,
            "Maximum target tokens must be greater than zero"
        );

        Self {
            max_source_tokens,
            max_target_tokens,
            source_token_counter,
            target_token_counter,
        }
    }

    pub fn segment_pair(&self, pair: &TranslationPair) -> Vec<TranslationSegment> {
        self.segment_pair_checked(pair)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn segment_pair_checked(
        &self,
        pair: &TranslationPair,
    ) -> Result<Vec<TranslationSegment>, String> {
        let source = pair.source.trim();
        let target = pair.target.trim();

        if source.is_empty() || target.is_empty() {
            return Ok(Vec::new());
        }

        let source_tokens = self.source_token_counter.count(source);
        let target_tokens = self.target_token_counter.count(target);

        if source_tokens <= self.max_source_tokens && target_tokens <= self.max_target_tokens {
            return Ok(vec![TranslationSegment {
                source: source.to_string(),
                target: target.to_string(),
            }]);
        }

        let source_sentences = split_sentences(source);
        let target_sentences = split_sentences(target);

        if source_sentences.len() != target_sentences.len() {
            return Err(format!(
                "Cannot safely segment translation pair: source and target sentence counts differ. Source sentences: {}, target sentences: {}, source tokens: {}, target tokens: {}, source limit: {}, target limit: {}",
                source_sentences.len(),
                target_sentences.len(),
                source_tokens,
                target_tokens,
                self.max_source_tokens,
                self.max_target_tokens
            ));
        }

        self.chunk_aligned_sentences(&source_sentences, &target_sentences)
    }

    fn chunk_aligned_sentences(
        &self,
        source_sentences: &[String],
        target_sentences: &[String],
    ) -> Result<Vec<TranslationSegment>, String> {
        let mut segments = Vec::new();
        let mut source_buffer = String::new();
        let mut target_buffer = String::new();

        for (index, (source_sentence, target_sentence)) in source_sentences
            .iter()
            .zip(target_sentences.iter())
            .enumerate()
        {
            let source_sentence_tokens = self.source_token_counter.count(source_sentence);

            let target_sentence_tokens = self.target_token_counter.count(target_sentence);

            if source_sentence_tokens > self.max_source_tokens
                || target_sentence_tokens > self.max_target_tokens
            {
                return Err(format!(
                    "Cannot safely segment oversized sentence pair at position {}. Source tokens: {}, target tokens: {}, source limit: {}, target limit: {}. The sentence pair must be split using a reliable alignment method or excluded.",
                    index + 1,
                    source_sentence_tokens,
                    target_sentence_tokens,
                    self.max_source_tokens,
                    self.max_target_tokens
                ));
            }

            let source_candidate = append_text(&source_buffer, source_sentence);

            let target_candidate = append_text(&target_buffer, target_sentence);

            let source_overflow =
                self.source_token_counter.count(&source_candidate) > self.max_source_tokens;

            let target_overflow =
                self.target_token_counter.count(&target_candidate) > self.max_target_tokens;

            if !source_buffer.is_empty() && (source_overflow || target_overflow) {
                push_segment(&source_buffer, &target_buffer, &mut segments);

                source_buffer = source_sentence.clone();
                target_buffer = target_sentence.clone();
            } else {
                source_buffer = source_candidate;
                target_buffer = target_candidate;
            }
        }

        push_segment(&source_buffer, &target_buffer, &mut segments);

        Ok(segments)
    }

    pub fn segment_dataset(&self, dataset: &[TranslationPair]) -> Vec<TranslationSegment> {
        self.segment_dataset_checked(dataset)
            .unwrap_or_else(|error| panic!("{error}"))
    }

    pub fn segment_dataset_checked(
        &self,
        dataset: &[TranslationPair],
    ) -> Result<Vec<TranslationSegment>, String> {
        let mut segments = Vec::new();

        for (index, pair) in dataset.iter().enumerate() {
            let pair_segments = self.segment_pair_checked(pair).map_err(|error| {
                format!("Failed to segment dataset pair {}: {}", index + 1, error)
            })?;

            segments.extend(pair_segments);
        }

        Ok(segments)
    }

    pub fn max_source_tokens(&self) -> usize {
        self.max_source_tokens
    }

    pub fn max_target_tokens(&self) -> usize {
        self.max_target_tokens
    }
}

impl DocumentSegmenter<WhitespaceTokenCounter, WhitespaceTokenCounter> {
    pub fn with_whitespace_counter(max_tokens: usize) -> Self {
        Self::new(
            max_tokens,
            max_tokens,
            WhitespaceTokenCounter,
            WhitespaceTokenCounter,
        )
    }
}

fn push_segment(source: &str, target: &str, out: &mut Vec<TranslationSegment>) {
    let source = source.trim();
    let target = target.trim();

    if source.is_empty() || target.is_empty() {
        return;
    }

    out.push(TranslationSegment {
        source: source.to_string(),
        target: target.to_string(),
    });
}

fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();

    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();

    for i in 0..len {
        let character = chars[i];

        current.push(character);

        if !matches!(character, '.' | '!' | '?' | '؟' | '۔' | '…') {
            continue;
        }

        let next = chars.get(i + 1).copied();

        let continuation = matches!(next, Some('.') | Some('!') | Some('?') | Some('…'));

        let decimal = character == '.'
            && i > 0
            && i + 1 < len
            && chars[i - 1].is_ascii_digit()
            && chars[i + 1].is_ascii_digit();

        let abbreviation = character == '.'
            && i > 0
            && i + 1 < len
            && chars[i - 1].is_ascii_alphabetic()
            && chars[i + 1].is_ascii_alphabetic();

        if continuation || decimal || abbreviation {
            continue;
        }

        let sentence = current.trim();

        if !sentence.is_empty() {
            sentences.push(sentence.to_string());
            current.clear();
        }
    }

    let remainder = current.trim();

    if !remainder.is_empty() {
        sentences.push(remainder.to_string());
    }

    sentences
}

fn append_text(current: &str, next: &str) -> String {
    if current.is_empty() {
        next.to_string()
    } else {
        format!("{current} {next}")
    }
}
