use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::softmax;


#[derive(Debug)]
pub struct QKV<B: Backend> {
    pub q: Tensor<B, 4>,
    pub k: Tensor<B, 4>,
    pub v: Tensor<B, 4>,
}


#[derive(Module, Debug)]
pub struct MaskedMHSA<B: Backend> {
    pub w_q: Linear<B>,
    pub w_k: Linear<B>,
    pub w_v: Linear<B>,
    pub w_o: Linear<B>,

    pub embedding_dimension: usize,
    pub num_heads: usize,
}

impl<B: Backend> MaskedMHSA<B> {
    /// Creates a new masked multi-head self-attention module.
    ///
    /// `embedding_dimension`
    ///     Total Transformer embedding dimension.
    ///
    /// `num_heads`
    ///     Number of attention heads.
    ///
    /// The embedding dimension must be divisible by
    /// the number of heads.
    pub fn new(embedding_dimension: usize, num_heads: usize, device: &B::Device, ) -> Self {
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

        let w_q = LinearConfig::new(embedding_dimension, embedding_dimension,).init(device);

        let w_k = LinearConfig::new(embedding_dimension,embedding_dimension,).init(device);

        let w_v = LinearConfig::new(embedding_dimension,embedding_dimension,).init(device);

        let w_o = LinearConfig::new(embedding_dimension,embedding_dimension,).init(device);

        Self {
            w_q,
            w_k,
            w_v,
            w_o,
            embedding_dimension,
            num_heads,
        }
    }

    /// Projects the input into Q, K, and V.
    ///
    /// Input:
    /// [batch, sequence, embedding]
    ///
    /// Output:
    /// Q: [batch, sequence, embedding]
    /// K: [batch, sequence, embedding]
    /// V: [batch, sequence, embedding]
    
    fn project_qkv(&self,input: Tensor<B, 3>,) -> (
        Tensor<B, 3>,
        Tensor<B, 3>,
        Tensor<B, 3>,
    ) {
        let q = self.w_q.forward(input.clone());
        let k = self.w_k.forward(input.clone());
        let v = self.w_v.forward(input);

        (q, k, v)
    }

    /// Splits the embedding dimension into attention heads.
    ///
    /// Before:
    ///
    /// [batch, sequence, embedding]
    ///
    /// After:
    ///
    /// [batch, heads, sequence, head_dimension]
    

    fn split_heads(
        &self,
        tensor: Tensor<B, 3>,
    ) -> Tensor<B, 4> {
        let [batch_size, sequence_length, embedding_dimension] = tensor.dims();

        let head_dimension = embedding_dimension / self.num_heads;

        tensor.reshape([
                batch_size,
                sequence_length,
                self.num_heads,
                head_dimension,
            ]).swap_dims(1, 2)
    }

    /// Combines attention heads.
    ///
    /// Before:
    ///
    /// [batch, heads, sequence, head_dimension]
    ///
    /// After:
    ///
    /// [batch, sequence, embedding]
    fn combine_heads(
        &self,
        tensor: Tensor<B, 4>,
    ) -> Tensor<B, 3> {
        let [batch_size, _, sequence_length, head_dimension] = tensor.dims();

        tensor.swap_dims(1, 2).reshape([
                batch_size,
                sequence_length,
                self.embedding_dimension,
            ])
    }

    /// Creates a causal attention mask.
    ///
    /// Allowed:
    ///
    /// position i → position j when j <= i
    ///
    /// Blocked:
    ///
    /// position i → position j when j > i
    fn causal_mask(&self, sequence_length: usize, device: &B::Device,) -> Tensor<B, 4> {
        let mut mask_data = vec![0.0_f32; sequence_length * sequence_length];

        for i in 0..sequence_length {
            for j in 0..sequence_length {
                let index =
                    i * sequence_length + j;

                if j > i {
                    // Future token.
                    //
                    // A large negative number causes
                    // softmax to effectively produce zero.
                    mask_data[index] = -1.0e9;
                }
            }
        }

        Tensor::<B, 1>::from_floats(mask_data.as_slice(),device,).reshape([1,1,sequence_length,sequence_length,])
    }

    /// Performs masked scaled dot-product attention.
    ///
    /// Q:
    /// [batch, heads, sequence, head_dimension]
    ///
    /// K:
    /// [batch, heads, sequence, head_dimension]
    ///
    /// V:
    /// [batch, heads, sequence, head_dimension]
    ///
    /// Returns:
    ///
    /// [batch, heads, sequence, head_dimension]
    fn attention( &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        device: &B::Device,
    ) -> Tensor<B, 4> {
        let head_dimension = self.embedding_dimension / self.num_heads;

        // K^T
        //
        // [B, H, S, D]
        //      ↓
        // [B, H, D, S]
        let k_transposed = k.swap_dims(2, 3);

        // QK^T
        //
        // [B, H, S, D]
        // ×
        // [B, H, D, S]
        //
        // =
        //
        // [B, H, S, S]
        let scores = q.matmul(k_transposed);

        // Scaled dot product.
        let scale = (head_dimension as f32).sqrt();

        let scores = scores / scale;

        // Create causal mask.
        let mask = self.causal_mask(scores.dims()[2], device, );

        // Prevent attention to future tokens.
        let scores = scores + mask;

        // Softmax over the last dimension.
        let attention_weights = softmax(scores, 3);

        // Attention × V
        //
        // [B, H, S, S]
        // ×
        // [B, H, S, D]
        //
        // =
        //
        // [B, H, S, D]
        attention_weights.matmul(v)
    }

    /// Forward pass.
    ///
    /// Input:
    ///
    /// [batch_size, sequence_length, embedding_dimension]
    ///
    /// Output:
    ///
    /// [batch_size, sequence_length, embedding_dimension]
    pub fn forward(
        &self,
        input: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let device = input.device();

        // -------------------------------------------------
        // 1. Q / K / V projections
        // -------------------------------------------------

        let (q, k, v) =
            self.project_qkv(input);

        // -------------------------------------------------
        // 2. Split into multiple attention heads
        // -------------------------------------------------

        let q = self.split_heads(q);
        let k = self.split_heads(k);
        let v = self.split_heads(v);

        // -------------------------------------------------
        // 3. Masked scaled dot-product attention
        // -------------------------------------------------

        let attention_output =
            self.attention(
                q,
                k,
                v,
                &device,
            );

        // -------------------------------------------------
        // 4. Concatenate heads
        // -------------------------------------------------

        let combined =
            self.combine_heads(attention_output);

        // -------------------------------------------------
        // 5. Output projection
        // -------------------------------------------------

        self.w_o.forward(combined)
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