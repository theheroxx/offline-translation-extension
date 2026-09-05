use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;


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

    pub embedding_dimension: usize,
    pub num_heads: usize,
}

impl<B: Backend> MHSA<B> {
    /// Creates a new Multi-Head Self-Attention module.
    ///
    /// `embedding_dimension`
    ///     Dimension of each token embedding.
    ///
    /// `num_heads`
    ///     Number of attention heads.
    pub fn new(embedding_dimension: usize, num_heads: usize, device: &B::Device,) -> Self {
        assert!(
            embedding_dimension > 0,
            "Embedding dimension must be greater than zero"
        );

        assert!(
            num_heads > 0,
            "Number of heads must be greater than zero"
        );

        assert_eq!(
            embedding_dimension % num_heads,
            0,
            "Embedding dimension ({}) must be divisible by number of heads ({})",
            embedding_dimension,
            num_heads
        );

        let w_q = LinearConfig::new( embedding_dimension, embedding_dimension,).init(device);

        let w_k = LinearConfig::new(embedding_dimension, embedding_dimension,).init(device);

        let w_v = LinearConfig::new( embedding_dimension, embedding_dimension,).init(device);

        Self {
            w_q,
            w_k,
            w_v,
            embedding_dimension,
            num_heads,
        }
    }

    /// Projects the input into Query, Key, and Value tensors.
    ///
    /// Input:
    ///
    /// [batch_size, sequence_length, embedding_dimension]
    ///
    /// Output:
    ///
    /// Q: [batch_size, sequence_length, embedding_dimension]
    /// K: [batch_size, sequence_length, embedding_dimension]
    /// V: [batch_size, sequence_length, embedding_dimension]
    pub fn project_qkv( &self,input: Tensor<B, 3>,) -> QKV<B> {
        let q = self.w_q.forward(input.clone());
        let k = self.w_k.forward(input.clone());
        let v = self.w_v.forward(input);

        QKV {
            q,
            k,
            v,
        }
    }

    /// Returns the dimension of each attention head.
    pub fn head_dimension(&self) -> usize {
        self.embedding_dimension / self.num_heads
    }

    /// Returns the number of attention heads.
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Returns the embedding dimension.
    pub fn embedding_dimension(&self) -> usize {
        self.embedding_dimension
    }
}