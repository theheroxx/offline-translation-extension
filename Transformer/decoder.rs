
use burn::module::Module;
use burn::nn::{LayerNorm, LayerNormConfig};
use burn::prelude::*;

use crate::Attention::cross_MHSA::CrossMHSA;
use crate::Attention::masked_MHSA::MaskedMHSA;
use crate::FFN::FFN;

#[derive(Module, Debug)]
pub struct DecoderBlock<B: Backend> {
    pub norm1: LayerNorm<B>,
    pub masked_attention: MaskedMHSA<B>,
    pub norm2: LayerNorm<B>,
    pub cross_attention: CrossMHSA<B>,
    pub norm3: LayerNorm<B>,
    pub ffn: FFN<B>,
}

impl<B: Backend> DecoderBlock<B> {
    pub fn new(
        embedding_dimension: usize,
        num_heads: usize,
        ffn_hidden_dimension: usize,
        max_encoder_sequence_length: usize,
        max_decoder_sequence_length: usize,
        device: &B::Device,
    ) -> Self {
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
            embedding_dimension % num_heads == 0,
            "Embedding dimension must be divisible by the number of attention heads"
        );

        let norm1 =
            LayerNormConfig::new(
                embedding_dimension
            )
            .init(device);

        let masked_attention =
            MaskedMHSA::new(
                embedding_dimension,
                num_heads,
                max_decoder_sequence_length,
                device,
            );

        let norm2 =
            LayerNormConfig::new(
                embedding_dimension
            )
            .init(device);

        let cross_attention =
            CrossMHSA::new(
                embedding_dimension,
                num_heads,
                max_decoder_sequence_length,
                max_encoder_sequence_length,
                device,
            );

        let norm3 =
            LayerNormConfig::new(
                embedding_dimension
            )
            .init(device);

        let ffn =
            FFN::new(
                embedding_dimension,
                ffn_hidden_dimension,
                embedding_dimension,
                device,
            );

        Self {
            norm1,
            masked_attention,
            norm2,
            cross_attention,
            norm3,
            ffn,
        }
    }

    pub fn forward(
        &self,
        decoder_input: Tensor<B, 3>,
        encoder_output: Tensor<B, 3>,
        target_padding_mask: Option<Tensor<B, 2, Bool>>,
        source_padding_mask: Option<Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 3> {
        let normalized_decoder =
            self.norm1.forward(
                decoder_input.clone()
            );

        let self_attention_output =
            self.masked_attention.forward(
                normalized_decoder,
                target_padding_mask,
            );

        let x =
            decoder_input +
            self_attention_output;

        let normalized_x =
            self.norm2.forward(
                x.clone()
            );

        let cross_attention_output =
            self.cross_attention.forward(
                normalized_x,
                encoder_output,
                source_padding_mask,
            );

        let x =
            x + cross_attention_output;

        let normalized_x =
            self.norm3.forward(
                x.clone()
            );

        let ffn_output =
            self.ffn.forward(
                normalized_x
            );

        x + ffn_output
    }

    pub fn embedding_dimension(&self) -> usize {
        self.masked_attention
            .embedding_dimension()
    }

    pub fn num_heads(&self) -> usize {
        self.masked_attention
            .num_heads()
    }
}
