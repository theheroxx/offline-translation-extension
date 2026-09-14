use crate::Tokenizer::dataset::TranslationPair;
use crate::Tokenizer::tokenizer::Tokenizer;

#[derive(Debug, Clone)]
pub struct TranslationBatch {
    pub source_ids: Vec<Vec<usize>>,
    pub target_input_ids: Vec<Vec<usize>>,
    pub target_label_ids: Vec<Vec<usize>>,
    pub source_lengths: Vec<usize>,
    pub target_lengths: Vec<usize>,
    pub max_source_length: usize,
    pub max_target_length: usize,
    pub pad_id: usize,
}

impl TranslationBatch {
    pub fn from_pairs(
        pairs: &[TranslationPair],
        source_tokenizer: &Tokenizer,
        target_tokenizer: &Tokenizer,
    ) -> Self {
        assert!(
            !pairs.is_empty(),
            "Translation batch cannot be empty"
        );

        let pad_id = source_tokenizer.pad_id();

        assert_eq!(
            pad_id,
            target_tokenizer.pad_id(),
            "Source and target PAD token IDs must match"
        );

        let mut source_ids = Vec::with_capacity(pairs.len());
        let mut target_input_ids = Vec::with_capacity(pairs.len());
        let mut target_label_ids = Vec::with_capacity(pairs.len());

        let mut source_lengths = Vec::with_capacity(pairs.len());
        let mut target_lengths = Vec::with_capacity(pairs.len());

        let mut max_source_length = 0;
        let mut max_target_length = 0;

        for pair in pairs {
            let source = source_tokenizer.encode_source(
                &pair.source
            );

            let target_input = target_tokenizer.encode_target_input(
                &pair.target
            );

            let target_labels = target_tokenizer.encode_target_labels(
                &pair.target
            );

            assert_eq!(
                target_input.len(),
                target_labels.len(),
                "Target input and target label lengths must match"
            );

            let source_length = source.len();
            let target_length = target_input.len();

            max_source_length =
                max_source_length.max(source_length);

            max_target_length =
                max_target_length.max(target_length);

            source_lengths.push(source_length);
            target_lengths.push(target_length);

            source_ids.push(source);
            target_input_ids.push(target_input);
            target_label_ids.push(target_labels);
        }

        for sequence in &mut source_ids {
            sequence.resize(
                max_source_length,
                pad_id,
            );
        }

        for sequence in &mut target_input_ids {
            sequence.resize(
                max_target_length,
                pad_id,
            );
        }

        for sequence in &mut target_label_ids {
            sequence.resize(
                max_target_length,
                pad_id,
            );
        }

        Self {
            source_ids,
            target_input_ids,
            target_label_ids,
            source_lengths,
            target_lengths,
            max_source_length,
            max_target_length,
            pad_id,
        }
    }

    pub fn batch_size(&self) -> usize {
        self.source_ids.len()
    }

    pub fn source_shape(&self) -> [usize; 2] {
        [
            self.batch_size(),
            self.max_source_length,
        ]
    }

    pub fn target_shape(&self) -> [usize; 2] {
        [
            self.batch_size(),
            self.max_target_length,
        ]
    }

    pub fn source_padding_mask(&self) -> Vec<Vec<bool>> {
        self.source_ids
            .iter()
            .map(|sequence| {
                sequence
                    .iter()
                    .map(|&token_id| token_id == self.pad_id)
                    .collect()
            })
            .collect()
    }

    pub fn target_padding_mask(&self) -> Vec<Vec<bool>> {
        self.target_input_ids
            .iter()
            .map(|sequence| {
                sequence
                    .iter()
                    .map(|&token_id| token_id == self.pad_id)
                    .collect()
            })
            .collect()
    }

    pub fn target_label_padding_mask(&self) -> Vec<Vec<bool>> {
        self.target_label_ids
            .iter()
            .map(|sequence| {
                sequence
                    .iter()
                    .map(|&token_id| token_id == self.pad_id)
                    .collect()
            })
            .collect()
    }

    pub fn source_ids_flattened(&self) -> Vec<usize> {
        self.source_ids
            .iter()
            .flat_map(|sequence| sequence.iter().copied())
            .collect()
    }

    pub fn target_input_ids_flattened(&self) -> Vec<usize> {
        self.target_input_ids
            .iter()
            .flat_map(|sequence| sequence.iter().copied())
            .collect()
    }

    pub fn target_label_ids_flattened(&self) -> Vec<usize> {
        self.target_label_ids
            .iter()
            .flat_map(|sequence| sequence.iter().copied())
            .collect()
    }
}
