use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::softmax;

use crate::PositionalEncoding::positional_encoding::PositionalEncoding;

#[derive(Module, Debug)]
pub struct MaskedMHSA<B: Backend> {
pub w_q: Linear<B>,
pub w_k: Linear<B>,
pub w_v: Linear<B>,
pub w_o: Linear<B>,
pub rope: PositionalEncoding<B>,
pub embedding_dimension: usize,
pub num_heads: usize,
}

impl<B: Backend> MaskedMHSA<B> {
pub fn new(
embedding_dimension: usize,
num_heads: usize,
max_sequence_length: usize,
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

    let head_dimension = embedding_dimension / num_heads;

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

    let rope = PositionalEncoding::new(
        head_dimension,
        max_sequence_length,
        10_000.0,
        device,
    );

    Self {
        w_q,
        w_k,
        w_v,
        w_o,
        rope,
        embedding_dimension,
        num_heads,
    }
}

fn project_qkv(
    &self,
    input: Tensor<B, 3>,
) -> (
    Tensor<B, 3>,
    Tensor<B, 3>,
    Tensor<B, 3>,
) {
    let q = self.w_q.forward(input.clone());
    let k = self.w_k.forward(input.clone());
    let v = self.w_v.forward(input);

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
        "Input embedding dimension does not match MaskedMHSA embedding dimension"
    );

    let head_dimension = self.head_dimension();

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
        "Number of attention heads does not match MaskedMHSA configuration"
    );

    assert_eq!(
        head_dimension,
        self.head_dimension(),
        "Attention head dimension does not match MaskedMHSA configuration"
    );

    tensor
        .swap_dims(1, 2)
        .reshape([
            batch_size,
            sequence_length,
            self.embedding_dimension,
        ])
}

fn causal_mask(
    &self,
    sequence_length: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    let mut mask_data =
        vec![0.0_f32; sequence_length * sequence_length];

    for i in 0..sequence_length {
        for j in 0..sequence_length {
            if j > i {
                mask_data[
                    i * sequence_length + j
                ] = -1.0e9;
            }
        }
    }

    Tensor::<B, 1>::from_floats(
        mask_data.as_slice(),
        device,
    )
    .reshape([
        1,
        1,
        sequence_length,
        sequence_length,
    ])
}

fn attention(
    &self,
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    device: &B::Device,
) -> Tensor<B, 4> {
    let head_dimension = self.head_dimension();

    let scores = q
        .matmul(k.swap_dims(2, 3))
        / (head_dimension as f32).sqrt();

    let sequence_length =
        scores.shape().dims::<4>()[2];

    let mask = self.causal_mask(
        sequence_length,
        device,
    );

    let scores = scores + mask;

    let attention_weights = softmax(scores, 3);

    attention_weights.matmul(v)
}

pub fn forward(
    &self,
    input: Tensor<B, 3>,
) -> Tensor<B, 3> {
    let device = input.device();

    let (q, k, v) =
        self.project_qkv(input);

    let q = self.rope.forward(
        self.split_heads(q)
    );

    let k = self.rope.forward(
        self.split_heads(k)
    );

    let v = self.split_heads(v);

    let attention_output =
        self.attention(
            q,
            k,
            v,
            &device,
        );

    let combined =
        self.combine_heads(
            attention_output,
        );

    self.w_o.forward(combined)
}

pub fn head_dimension(&self) -> usize {
    self.embedding_dimension / self.num_heads
}

pub fn num_heads(&self) -> usize {
    self.num_heads
}

pub fn embedding_dimension(&self) -> usize {
    self.embedding_dimension
}

}
