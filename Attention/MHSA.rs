use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::softmax;

use crate::PositionalEncoding::positional_encoding::PositionalEncoding;

#[derive(Debug)]
pub struct QKV<B: Backend> {
pub q: Tensor<B, 3>,
pub k: Tensor<B, 3>,
pub v: Tensor<B, 3>,
}

#[derive(Module, Debug)]
pub struct MHSA<B: Backend> {
pub w_q: Linear<B>,
pub w_k: Linear<B>,
pub w_v: Linear<B>,
pub w_o: Linear<B>,
pub rope: PositionalEncoding<B>,
pub embedding_dimension: usize,
pub num_heads: usize,
}

impl<B: Backend> MHSA<B> {
pub fn new(
embedding_dimension: usize,
num_heads: usize,
max_sequence_length: usize,
device: &B::Device,
) -> Self {
assert!(embedding_dimension > 0);
assert!(num_heads > 0);

    assert!(
        embedding_dimension % num_heads == 0,
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

pub fn project_qkv(&self, input: Tensor<B, 3>) -> QKV<B> {
    let q = self.w_q.forward(input.clone());
    let k = self.w_k.forward(input.clone());
    let v = self.w_v.forward(input);

    QKV { q, k, v }
}

fn split_heads(&self, tensor: Tensor<B, 3>) -> Tensor<B, 4> {
    let [batch_size, sequence_length, embedding_dimension] =
        tensor.shape().dims::<3>();

    assert_eq!(
        embedding_dimension,
        self.embedding_dimension,
        "Input embedding dimension does not match MHSA embedding dimension"
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

fn combine_heads(&self, tensor: Tensor<B, 4>) -> Tensor<B, 3> {
    let [
        batch_size,
        num_heads,
        sequence_length,
        head_dimension,
    ] = tensor.shape().dims::<4>();

    assert_eq!(
        num_heads,
        self.num_heads,
        "Number of attention heads does not match MHSA configuration"
    );

    assert_eq!(
        head_dimension,
        self.head_dimension(),
        "Attention head dimension does not match MHSA configuration"
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
) -> Tensor<B, 4> {
    let head_dimension = self.head_dimension();

    let scores = q
        .matmul(k.swap_dims(2, 3))
        / (head_dimension as f32).sqrt();

    let attention_weights = softmax(scores, 3);

    attention_weights.matmul(v)
}

pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
    let qkv = self.project_qkv(input);

    let q = self.rope.forward(
        self.split_heads(qkv.q)
    );

    let k = self.rope.forward(
        self.split_heads(qkv.k)
    );

    let v = self.split_heads(qkv.v);

    let attention_output = self.attention(q, k, v);

    let combined = self.combine_heads(attention_output);

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
