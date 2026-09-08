use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::softmax;


#[derive(Debug)]
pub struct QKV<B: Backend> {
    pub q: Tensor<B, 3>,
    pub k: Tensor<B, 3>,
    pub v: Tensor<B, 3>,
}


#[derive(Module, Debug)]
pub struct MHSA<B: Backend> {
    /// Query projection.
    pub w_q: Linear<B>,

    /// Key projection.
    pub w_k: Linear<B>,

    /// Value projection.
    pub w_v: Linear<B>,

    /// Output projection.
    pub w_o: Linear<B>,

    /// Total embedding dimension.
    pub embedding_dimension: usize,

    /// Number of attention heads.
    pub num_heads: usize,
}

impl<B: Backend> MHSA<B> {

    pub fn new(
        embedding_dimension: usize,
        num_heads: usize,
        device: &B::Device,
    ) -> Self {
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

        let w_q =
            LinearConfig::new(
                embedding_dimension,
                embedding_dimension,
            )
            .init(device);

        let w_k =
            LinearConfig::new(
                embedding_dimension,
                embedding_dimension,
            )
            .init(device);

        let w_v =
            LinearConfig::new(
                embedding_dimension,
                embedding_dimension,
            )
            .init(device);

        let w_o =
            LinearConfig::new(
                embedding_dimension,
                embedding_dimension,
            )
            .init(device);

        Self {
            w_q,
            w_k,
            w_v,
            w_o,
            embedding_dimension,
            num_heads,
        }
    }

    /// Projects the input sequence into Q, K and V.
    ///
    /// Input:
    ///
    /// [batch, sequence_length, embedding_dimension]
    ///
    /// Because this is SELF-attention, the same input
    /// sequence is used for all three projections.
    ///
    /// Q = X Wq
    /// K = X Wk
    /// V = X Wv
    pub fn project_qkv(
        &self,
        input: Tensor<B, 3>,
    ) -> QKV<B> {
        let q = self.w_q.forward(input.clone());

        let k = self.w_k.forward(input.clone());

        let v = self.w_v.forward(input);

        QKV { q, k, v }
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
    ///
    /// Example:
    ///
    /// [2, 10, 512]
    ///
    /// with 8 heads becomes:
    ///
    /// [2, 8, 10, 64]
    fn split_heads(
        &self,
        tensor: Tensor<B, 3>,
    ) -> Tensor<B, 4> {
        let [
            batch_size,
            sequence_length,
            embedding_dimension,
        ] = tensor.dims();

        let head_dimension =
            embedding_dimension / self.num_heads;

        tensor
            .reshape([
                batch_size,
                sequence_length,
                self.num_heads,
                head_dimension,
            ])
            .swap_dims(1, 2)
    }

    /// Combines all attention heads.
    ///
    /// Before:
    ///
    /// [batch, heads, sequence, head_dimension]
    ///
    /// After:
    ///
    /// [batch, sequence, embedding_dimension]
    fn combine_heads(
        &self,
        tensor: Tensor<B, 4>,
    ) -> Tensor<B, 3> {
        let [
            batch_size,
            _,
            sequence_length,
            head_dimension,
        ] = tensor.dims();

        tensor
            .swap_dims(1, 2)
            .reshape([
                batch_size,
                sequence_length,
                head_dimension * self.num_heads,
            ])
    }

    /// Performs scaled dot-product self-attention.
    ///
    /// Q:
    ///
    /// [batch, heads, sequence, head_dimension]
    ///
    /// K:
    ///
    /// [batch, heads, sequence, head_dimension]
    ///
    /// V:
    ///
    /// [batch, heads, sequence, head_dimension]
    ///
    /// First:
    ///
    /// QKᵀ
    ///
    /// produces:
    ///
    /// [batch, heads, sequence, sequence]
    ///
    /// Then:
    ///
    /// QKᵀ / sqrt(head_dimension)
    ///
    /// followed by softmax.
    fn attention(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let head_dimension =
            self.embedding_dimension / self.num_heads;

        // K:
        //
        // [B, H, S, D]
        //
        // becomes:
        //
        // [B, H, D, S]
        let k_transposed =
            k.swap_dims(2, 3);

        // QKᵀ
        //
        // [B, H, S, D]
        //
        // ×
        //
        // [B, H, D, S]
        //
        // =
        //
        // [B, H, S, S]
        let scores =
            q.matmul(k_transposed);

        // Scale attention scores.
        //
        // This prevents the dot products from becoming
        // excessively large when the head dimension grows.
        let scale =
            (head_dimension as f32).sqrt();

        let scores =
            scores / scale;

        // Convert scores into attention probabilities.
        //
        // Softmax is applied over the KEY sequence dimension.
        let attention_weights =
            softmax(scores, 3);

        // Weighted sum of values.
        //
        // [B, H, S, S]
        //
        // ×
        //
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
    /// [batch, sequence_length, embedding_dimension]
    ///
    /// Output:
    ///
    /// [batch, sequence_length, embedding_dimension]
    pub fn forward(
        &self,
        input: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        // ---------------------------------------------------------
        // 1. Create Q, K and V
        // ---------------------------------------------------------
        let qkv =
            self.project_qkv(input);

        // ---------------------------------------------------------
        // 2. Split Q, K and V into multiple heads
        // ---------------------------------------------------------
        let q =
            self.split_heads(qkv.q);

        let k =
            self.split_heads(qkv.k);

        let v =
            self.split_heads(qkv.v);

        // ---------------------------------------------------------
        // 3. Calculate attention
        // ---------------------------------------------------------
        let attention_output =
            self.attention(q, k, v);

        // ---------------------------------------------------------
        // 4. Combine the heads
        // ---------------------------------------------------------
        let combined =
            self.combine_heads(attention_output);

        // ---------------------------------------------------------
        // 5. Final output projection
        // ---------------------------------------------------------
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

    /// Returns the total embedding dimension.
    pub fn embedding_dimension(&self) -> usize {
        self.embedding_dimension
    }
}
