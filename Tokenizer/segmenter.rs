
use crate::Tokenizer::dataset::TranslationPair;

#[derive(Debug, Clone)]
pub struct TranslationSegment {
    pub source: String,
    pub target: String,
}

pub trait TokenCounter {
    fn count(&self, text: &str) -> usize;
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
        let source_paragraphs = split_paragraphs(&pair.source);
        let target_paragraphs = split_paragraphs(&pair.target);

        assert_eq!(
            source_paragraphs.len(),
            target_paragraphs.len(),
            "Source and target paragraph counts must match (source={}, target={})",
            source_paragraphs.len(),
            target_paragraphs.len()
        );

        let mut segments = Vec::new();

        for (source_paragraph, target_paragraph) in
            source_paragraphs.iter().zip(target_paragraphs.iter())
        {
            self.segment_paragraph(
                source_paragraph,
                target_paragraph,
                &mut segments,
            );
        }

        segments
    }

    fn segment_paragraph(
        &self,
        source_paragraph: &str,
        target_paragraph: &str,
        out: &mut Vec<TranslationSegment>,
    ) {
        let source_tokens = self.source_token_counter.count(source_paragraph);
        let target_tokens = self.target_token_counter.count(target_paragraph);

        if source_tokens <= self.max_source_tokens
            && target_tokens <= self.max_target_tokens
        {
            push_segment(
                source_paragraph,
                target_paragraph,
                out,
            );
            return;
        }

        let source_sentences = split_sentences(source_paragraph);
        let target_sentences = split_sentences(target_paragraph);

        if source_sentences.len() == target_sentences.len() {
            self.chunk_aligned_sentences(
                &source_sentences,
                &target_sentences,
                out,
            );
            return;
        }

        self.handle_unaligned_paragraph(
            source_paragraph,
            target_paragraph,
            out,
        );
    }

    fn chunk_aligned_sentences(
        &self,
        source_sentences: &[String],
        target_sentences: &[String],
        out: &mut Vec<TranslationSegment>,
    ) {
        let mut source_buffer = String::new();
        let mut target_buffer = String::new();

        for (source_sentence, target_sentence) in
            source_sentences.iter().zip(target_sentences.iter())
        {
            let source_sentence_tokens =
                self.source_token_counter.count(source_sentence);

            let target_sentence_tokens =
                self.target_token_counter.count(target_sentence);

            if source_sentence_tokens > self.max_source_tokens
                || target_sentence_tokens > self.max_target_tokens
            {
                push_segment(
                    &source_buffer,
                    &target_buffer,
                    out,
                );

                source_buffer.clear();
                target_buffer.clear();

                self.split_aligned_oversized_sentence(
                    source_sentence,
                    target_sentence,
                    out,
                );

                continue;
            }

            let source_candidate =
                append_text(&source_buffer, source_sentence);

            let target_candidate =
                append_text(&target_buffer, target_sentence);

            let source_overflow =
                self.source_token_counter.count(&source_candidate)
                    > self.max_source_tokens;

            let target_overflow =
                self.target_token_counter.count(&target_candidate)
                    > self.max_target_tokens;

            if !source_buffer.is_empty()
                && (source_overflow || target_overflow)
            {
                push_segment(
                    &source_buffer,
                    &target_buffer,
                    out,
                );

                source_buffer = source_sentence.clone();
                target_buffer = target_sentence.clone();
            } else {
                source_buffer = source_candidate;
                target_buffer = target_candidate;
            }
        }

        push_segment(
            &source_buffer,
            &target_buffer,
            out,
        );
    }

    fn split_aligned_oversized_sentence(
        &self,
        source_sentence: &str,
        target_sentence: &str,
        out: &mut Vec<TranslationSegment>,
    ) {
        let source_words: Vec<&str> =
            source_sentence.split_whitespace().collect();

        let target_words: Vec<&str> =
            target_sentence.split_whitespace().collect();

        if source_words.is_empty() || target_words.is_empty() {
            return;
        }

        let source_ratio =
            source_words.len() as f64 / target_words.len() as f64;

        let mut source_start = 0usize;
        let mut target_start = 0usize;

        while source_start < source_words.len()
            || target_start < target_words.len()
        {
            let remaining_source =
                source_words.len().saturating_sub(source_start);

            let remaining_target =
                target_words.len().saturating_sub(target_start);

            if remaining_source == 0 || remaining_target == 0 {
                let source =
                    source_words[source_start..].join(" ");

                let target =
                    target_words[target_start..].join(" ");

                if !source.is_empty() && !target.is_empty() {
                    push_segment(&source, &target, out);
                }

                break;
            }

            let mut source_end =
                (source_start + self.max_source_tokens)
                    .min(source_words.len());

            let estimated_target_end =
                target_start
                    + ((source_end - source_start) as f64 / source_ratio)
                        .ceil() as usize;

            let mut target_end =
                estimated_target_end.min(target_words.len());

            loop {
                let source =
                    source_words[source_start..source_end].join(" ");

                let target =
                    target_words[target_start..target_end].join(" ");

                let source_fits =
                    self.source_token_counter.count(&source)
                        <= self.max_source_tokens;

                let target_fits =
                    self.target_token_counter.count(&target)
                        <= self.max_target_tokens;

                if source_fits && target_fits {
                    push_segment(&source, &target, out);

                    source_start = source_end;
                    target_start = target_end;

                    break;
                }

                if source_end > source_start + 1 {
                    source_end -= 1;
                    continue;
                }

                if target_end > target_start + 1 {
                    target_end -= 1;
                    continue;
                }

                let source =
                    source_words[source_start..source_end].join(" ");

                let target =
                    target_words[target_start..target_end].join(" ");

                push_segment(&source, &target, out);

                source_start = source_end;
                target_start = target_end;

                break;
            }
        }
    }

    fn handle_unaligned_paragraph(
        &self,
        source_paragraph: &str,
        target_paragraph: &str,
        out: &mut Vec<TranslationSegment>,
    ) {
        let source_tokens =
            self.source_token_counter.count(source_paragraph);

        let target_tokens =
            self.target_token_counter.count(target_paragraph);

        if source_tokens <= self.max_source_tokens
            && target_tokens <= self.max_target_tokens
        {
            push_segment(
                source_paragraph,
                target_paragraph,
                out,
            );
            return;
        }

        panic!(
            "Cannot safely segment an oversized paragraph because source and target sentence boundaries do not align. \
             Source tokens: {}, target tokens: {}, max source: {}, max target: {}",
            source_tokens,
            target_tokens,
            self.max_source_tokens,
            self.max_target_tokens
        );
    }

    pub fn segment_dataset(
        &self,
        dataset: &[TranslationPair],
    ) -> Vec<TranslationSegment> {
        let mut segments = Vec::new();

        for pair in dataset {
            segments.extend(self.segment_pair(pair));
        }

        segments
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

fn push_segment(
    source: &str,
    target: &str,
    out: &mut Vec<TranslationSegment>,
) {
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

fn split_paragraphs(text: &str) -> Vec<String> {
    text.split("\n\n")
        .map(str::trim)
        .filter(|paragraph| !paragraph.is_empty())
        .map(String::from)
        .collect()
}

fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();

    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();

    for i in 0..len {
        let character = chars[i];

        current.push(character);

        if matches!(
            character,
            '.' | '!' | '?' | '؟' | '۔' | '…'
        ) {
            let next = chars.get(i + 1).copied();

            let continuation =
                matches!(
                    next,
                    Some('.') | Some('!') | Some('?') | Some('…')
                );

            let decimal =
                character == '.'
                    && i > 0
                    && i + 1 < len
                    && chars[i - 1].is_ascii_digit()
                    && chars[i + 1].is_ascii_digit();

            if continuation || decimal {
                continue;
            }

            let sentence = current.trim();

            if !sentence.is_empty() {
                sentences.push(sentence.to_string());
                current.clear();
            }
        }
    }

    let remainder = current.trim();

    if !remainder.is_empty() {
        sentences.push(remainder.to_string());
    }

    sentences
}

fn append_text(
    current: &str,
    next: &str,
) -> String {
    if current.is_empty() {
        next.to_string()
    } else {
        format!("{current} {next}")
    }
}

