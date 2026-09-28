use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::softmax;

use crate::PositionalEncoding::positional_encoding::PositionalEncoding;

#[derive(Module, Debug)]
pub struct CrossMHSA<B: Backend> {
    pub w_q: Linear<B>,
    pub w_k: Linear<B>,
    pub w_v: Linear<B>,
    pub w_o: Linear<B>,
    pub query_rope: PositionalEncoding<B>,
    pub key_rope: PositionalEncoding<B>,
    pub embedding_dimension: usize,
    pub num_heads: usize,
}

impl<B: Backend> CrossMHSA<B> {
    pub fn new(
        embedding_dimension: usize,
        num_heads: usize,
        max_decoder_sequence_length: usize,
        max_encoder_sequence_length: usize,
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

        assert_eq!(
            embedding_dimension % num_heads,
            0,
            "Embedding dimension ({}) must be divisible by number of heads ({})",
            embedding_dimension,
            num_heads
        );

        let head_dimension =
            embedding_dimension / num_heads;

        assert!(
            head_dimension % 2 == 0,
            "Attention head dimension ({}) must be even for RoPE",
            head_dimension
        );

        let w_q = LinearConfig::new(
            embedding_dimension,
            embedding_dimension,
        )
        .init(device);

        let w_k = LinearConfig::new(
            embedding_dimension,
            embedding_dimension,
        )
        .init(device);

        let w_v = LinearConfig::new(
            embedding_dimension,
            embedding_dimension,
        )
        .init(device);

        let w_o = LinearConfig::new(
            embedding_dimension,
            embedding_dimension,
        )
        .init(device);

        let query_rope = PositionalEncoding::new(
            head_dimension,
            max_decoder_sequence_length,
            10_000.0,
            device,
        );

        let key_rope = PositionalEncoding::new(
            head_dimension,
            max_encoder_sequence_length,
            10_000.0,
            device,
        );

        Self {
            w_q,
            w_k,
            w_v,
            w_o,
            query_rope,
            key_rope,
            embedding_dimension,
            num_heads,
        }
    }

    fn project_qkv(
        &self,
        decoder_input: Tensor<B, 3>,
        encoder_input: Tensor<B, 3>,
    ) -> (
        Tensor<B, 3>,
        Tensor<B, 3>,
        Tensor<B, 3>,
    ) {
        let q = self.w_q.forward(decoder_input);

        let k = self.w_k.forward(
            encoder_input.clone(),
        );

        let v = self.w_v.forward(
            encoder_input,
        );

        (q, k, v)
    }

    fn split_heads(
        &self,
        tensor: Tensor<B, 3>,
    ) -> Tensor<B, 4> {
        let [
            batch_size,
            sequence_length,
            embedding_dimension,
        ] = tensor.shape().dims::<3>();

        assert_eq!(
            embedding_dimension,
            self.embedding_dimension,
            "Input embedding dimension does not match CrossMHSA embedding dimension"
        );

        let head_dimension =
            self.head_dimension();

        tensor
            .reshape([
                batch_size,
                sequence_length,
                self.num_heads,
                head_dimension,
            ])
            .swap_dims(1, 2)
    }

    fn combine_heads(
        &self,
        tensor: Tensor<B, 4>,
    ) -> Tensor<B, 3> {
        let [
            batch_size,
            num_heads,
            sequence_length,
            head_dimension,
        ] = tensor.shape().dims::<4>();

        assert_eq!(
            num_heads,
            self.num_heads,
            "Number of attention heads does not match CrossMHSA configuration"
        );

        assert_eq!(
            head_dimension,
            self.head_dimension(),
            "Attention head dimension does not match CrossMHSA configuration"
        );

        tensor
            .swap_dims(1, 2)
            .reshape([
                batch_size,
                sequence_length,
                self.embedding_dimension,
            ])
    }

    fn attention(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        encoder_padding_mask: Option<Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 4> {
        let head_dimension =
            self.head_dimension();

        let mut scores = q
            .matmul(k.swap_dims(2, 3))
            / (head_dimension as f32).sqrt();

        if let Some(mask) =
            encoder_padding_mask
        {
            let [
                mask_batch_size,
                mask_sequence_length,
            ] = mask.shape().dims::<2>();

            let [
                score_batch_size,
                score_num_heads,
                query_length,
                key_length,
            ] = scores.shape().dims::<4>();

            assert_eq!(
                mask_batch_size,
                score_batch_size,
                "Encoder padding mask batch size does not match attention batch size"
            );

            assert_eq!(
                mask_sequence_length,
                key_length,
                "Encoder padding mask sequence length does not match encoder sequence length"
            );

            let mask = mask
                .reshape([
                    mask_batch_size,
                    1,
                    1,
                    mask_sequence_length,
                ])
                .repeat_dim(
                    1,
                    score_num_heads,
                )
                .repeat_dim(
                    2,
                    query_length,
                );

            let mask_value =
                scores.clone().full_like(-1.0e9);

            scores = scores.mask_where(
                mask,
                mask_value,
            );
        }

        let attention_weights =
            softmax(scores, 3);

        attention_weights.matmul(v)
    }

    pub fn forward(
        &self,
        decoder_input: Tensor<B, 3>,
        encoder_input: Tensor<B, 3>,
        encoder_padding_mask: Option<Tensor<B, 2, Bool>>,
    ) -> Tensor<B, 3> {
        let (q, k, v) =
            self.project_qkv(
                decoder_input,
                encoder_input,
            );

        let q = self.query_rope.forward(
            self.split_heads(q),
        );

        let k = self.key_rope.forward(
            self.split_heads(k),
        );

        let v = self.split_heads(v);

        let attention_output =
            self.attention(
                q,
                k,
                v,
                encoder_padding_mask,
            );

        let combined =
            self.combine_heads(
                attention_output,
            );

        self.w_o.forward(combined)
    }

    pub fn head_dimension(&self) -> usize {
        self.embedding_dimension
            / self.num_heads
    }

    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    pub fn embedding_dimension(&self) -> usize {
        self.embedding_dimension
    }
}
