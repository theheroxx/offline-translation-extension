
use burn::module::Module;
use burn::prelude::*;

use crate::Transformer::decoder::DecoderBlock;
use crate::Transformer::encoder::EncoderBlock;
use crate::Transformer::lm_head::LMHead;

#[derive(Module, Debug)]
pub struct Transformer<B: Backend> {
    pub encoder_layers: Vec<EncoderBlock<B>>,
    pub decoder_layers: Vec<DecoderBlock<B>>,
    pub lm_head: LMHead<B>,
    pub num_encoder_layers: usize,
    pub num_decoder_layers: usize,
    pub embedding_dimension: usize,
    pub num_heads: usize,
    pub ffn_hidden_dimension: usize,
    pub max_encoder_sequence_length: usize,
    pub max_decoder_sequence_length: usize,
    pub target_vocab_size: usize,
}

impl<B: Backend> Transformer<B> {
    pub fn new(
        num_encoder_layers: usize,
        num_decoder_layers: usize,
        embedding_dimension: usize,
        num_heads: usize,
        ffn_hidden_dimension: usize,
        max_encoder_sequence_length: usize,
        max_decoder_sequence_length: usize,
        target_vocab_size: usize,
        device: &B::Device,
    ) -> Self {
        assert!(
            num_encoder_layers > 0,
            "Number of encoder layers must be greater than zero"
        );

        assert!(
            num_decoder_layers > 0,
            "Number of decoder layers must be greater than zero"
        );

        assert!(
            embedding_dimension > 0,
            "Embedding dimension must be greater than zero"
        );

        assert!(
            num_heads > 0,
            "Number of attention heads must be greater than zero"
        );

        assert!(
            ffn_hidden_dimension > 0,
            "FFN hidden dimension must be greater than zero"
        );

        assert!(
            max_encoder_sequence_length > 0,
            "Maximum encoder sequence length must be greater than zero"
        );

        assert!(
            max_decoder_sequence_length > 0,
            "Maximum decoder sequence length must be greater than zero"
        );

        assert!(
            target_vocab_size > 0,
            "Target vocabulary size must be greater than zero"
        );

        assert!(
            embedding_dimension % num_heads == 0,
            "Embedding dimension must be divisible by number of attention heads"
        );

        let mut encoder_layers =
            Vec::with_capacity(num_encoder_layers);

        for _ in 0..num_encoder_layers {
            encoder_layers.push(
                EncoderBlock::new(
                    embedding_dimension,
                    num_heads,
                    ffn_hidden_dimension,
                    max_encoder_sequence_length,
                    device,
                )
            );
        }

        let mut decoder_layers =
            Vec::with_capacity(num_decoder_layers);

        for _ in 0..num_decoder_layers {
            decoder_layers.push(
                DecoderBlock::new(
                    embedding_dimension,
                    num_heads,
                    ffn_hidden_dimension,
                    max_encoder_sequence_length,
                    max_decoder_sequence_length,
                    device,
                )
            );
        }

        let lm_head =
            LMHead::new(
                embedding_dimension,
                target_vocab_size,
                device,
            );

        Self {
            encoder_layers,
            decoder_layers,
            lm_head,
            num_encoder_layers,
            num_decoder_layers,
            embedding_dimension,
            num_heads,
            ffn_hidden_dimension,
            max_encoder_sequence_length,
            max_decoder_sequence_length,
            target_vocab_size,
        }
    }

    pub fn encode(
        &self,
        input: Tensor<B, 3>,
        source_padding_mask: Option<Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 3> {
        let mut output = input;

        for layer in &self.encoder_layers {
            output = layer.forward(
                output,
                source_padding_mask.clone(),
            );
        }

        output
    }

    pub fn decode(
        &self,
        target: Tensor<B, 3>,
        encoder_output: Tensor<B, 3>,
        target_padding_mask: Option<Tensor<B, 2, Bool>>,
        source_padding_mask: Option<Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 3> {
        let mut output = target;

        for layer in &self.decoder_layers {
            output = layer.forward(
                output,
                encoder_output.clone(),
                target_padding_mask.clone(),
                source_padding_mask.clone(),
            );
        }

        output
    }

    pub fn forward(
        &self,
        source: Tensor<B, 3>,
        target: Tensor<B, 3>,
        source_padding_mask: Option<Tensor<B, 2, Bool>>,
        target_padding_mask: Option<Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 3> {
        let encoder_output =
            self.encode(
                source,
                source_padding_mask.clone(),
            );

        let decoder_output =
            self.decode(
                target,
                encoder_output,
                target_padding_mask,
                source_padding_mask,
            );

        self.lm_head.forward(
            decoder_output
        )
    }

    pub fn embedding_dimension(&self) -> usize {
        self.embedding_dimension
    }

    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    pub fn num_encoder_layers(&self) -> usize {
        self.num_encoder_layers
    }

    pub fn num_decoder_layers(&self) -> usize {
        self.num_decoder_layers
    }

    pub fn ffn_hidden_dimension(&self) -> usize {
        self.ffn_hidden_dimension
    }

    pub fn max_encoder_sequence_length(&self) -> usize {
        self.max_encoder_sequence_length
    }

    pub fn max_decoder_sequence_length(&self) -> usize {
        self.max_decoder_sequence_length
    }

    pub fn target_vocab_size(&self) -> usize {
        self.target_vocab_size
    }
}

