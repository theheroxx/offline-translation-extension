use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::softmax;

#[derive(Module, Debug)]
pub struct CrossMHSA<B: Backend> {
    /// Query projection.
    ///
    /// Decoder -> Q
    pub w_q: Linear<B>,

    /// Key projection.
    ///
    /// Encoder -> K
    pub w_k: Linear<B>,

    /// Value projection.
    ///
    /// Encoder -> V
    pub w_v: Linear<B>,

    /// Output projection.
    pub w_o: Linear<B>,

    /// Total Transformer embedding dimension.
    pub embedding_dimension: usize,

    /// Number of attention heads.
    pub num_heads: usize,
}

impl<B: Backend> CrossMHSA<B> {

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
            "Number of attention heads must be greater than zero"
        );

        assert_eq!(
            embedding_dimension % num_heads,
            0,
            "Embedding dimension ({}) must be divisible by number of heads ({})",
            embedding_dimension,
            num_heads
        );

        // ---------------------------------------------------------
        // Query projection
        // ---------------------------------------------------------

        let w_q = LinearConfig::new(
            embedding_dimension,
            embedding_dimension,
        )
        .init(device);

        // ---------------------------------------------------------
        // Key projection
        // ---------------------------------------------------------

        let w_k = LinearConfig::new(
            embedding_dimension,
            embedding_dimension,
        )
        .init(device);

        // ---------------------------------------------------------
        // Value projection
        // ---------------------------------------------------------

        let w_v = LinearConfig::new(
            embedding_dimension,
            embedding_dimension,
        )
        .init(device);

        // ---------------------------------------------------------
        // Output projection
        // ---------------------------------------------------------

        let w_o = LinearConfig::new(
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

    /// Projects decoder and encoder representations into Q, K and V.
    ///
    /// Decoder input:
    ///
    /// [batch, decoder_sequence, embedding]
    ///
    /// Encoder input:
    ///
    /// [batch, encoder_sequence, embedding]
    ///
    /// Q:
    ///
    /// [batch, decoder_sequence, embedding]
    ///
    /// K:
    ///
    /// [batch, encoder_sequence, embedding]
    ///
    /// V:
    ///
    /// [batch, encoder_sequence, embedding]
    ///
    /// The important distinction from self-attention is:
    ///
    /// Q = decoder * Wq
    /// K = encoder  * Wk
    /// V = encoder  * Wv
    fn project_qkv(
        &self,
        decoder_input: Tensor<B, 3>,
        encoder_input: Tensor<B, 3>,
    ) -> (
        Tensor<B, 3>,
        Tensor<B, 3>,
        Tensor<B, 3>,
    ) {
        // Decoder generates Q.
        let q =
            self.w_q.forward(decoder_input);

        // Encoder generates K and V.
        let k =
            self.w_k.forward(encoder_input.clone());

        let v =
            self.w_v.forward(encoder_input);

        (q, k, v)
    }

    /// Splits the embedding dimension into multiple attention heads.
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
    /// with 8 heads:
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

    /// Combines the attention heads back together.
    ///
    /// Before:
    ///
    /// [batch, heads, sequence, head_dimension]
    ///
    /// After:
    ///
    /// [batch, sequence, embedding]
    ///
    /// Notice that this method works with the decoder
    /// sequence length because the attention output has
    /// one representation for every decoder query.
    fn combine_heads(
        &self,
        tensor: Tensor<B, 4>,
    ) -> Tensor<B, 3> {
        let [
            batch_size,
            _num_heads,
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

    /// Performs scaled dot-product cross-attention.
    ///
    /// Q:
    ///
    /// [batch, heads, decoder_sequence, head_dimension]
    ///
    /// K:
    ///
    /// [batch, heads, encoder_sequence, head_dimension]
    ///
    /// V:
    ///
    /// [batch, heads, encoder_sequence, head_dimension]
    ///
    /// QK^T:
    ///
    /// [batch, heads, decoder_sequence, encoder_sequence]
    ///
    /// Output:
    ///
    /// [batch, heads, decoder_sequence, head_dimension]
    ///
    /// There is NO causal mask here.
    ///
    /// Every decoder position is allowed to attend to every
    /// encoder position.
    fn attention(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let head_dimension =
            self.embedding_dimension / self.num_heads;

        // ---------------------------------------------------------
        // 1. Transpose K
        // ---------------------------------------------------------
        //
        // K:
        //
        // [B, H, encoder_sequence, D]
        //
        // becomes:
        //
        // [B, H, D, encoder_sequence]

        let k_transposed =
            k.swap_dims(2, 3);

        // ---------------------------------------------------------
        // 2. Calculate QK^T
        // ---------------------------------------------------------
        //
        // Q:
        //
        // [B, H, decoder_sequence, D]
        //
        // K^T:
        //
        // [B, H, D, encoder_sequence]
        //
        // Result:
        //
        // [B, H, decoder_sequence, encoder_sequence]

        let scores =
            q.matmul(k_transposed);

        // ---------------------------------------------------------
        // 3. Scale the scores
        // ---------------------------------------------------------

        let scale =
            (head_dimension as f32).sqrt();

        let scores =
            scores / scale;

        // ---------------------------------------------------------
        // 4. Softmax
        // ---------------------------------------------------------
        //
        // Softmax is applied over the encoder sequence.
        //
        // Each decoder token therefore gets a probability
        // distribution over all encoder tokens.

        let attention_weights =
            softmax(scores, 3);

        // ---------------------------------------------------------
        // 5. Weighted sum of V
        // ---------------------------------------------------------
        //
        // Attention weights:
        //
        // [B, H, decoder_sequence, encoder_sequence]
        //
        // V:
        //
        // [B, H, encoder_sequence, D]
        //
        // Result:
        //
        // [B, H, decoder_sequence, D]

        attention_weights.matmul(v)
    }

    /// Forward pass for cross-attention.
    ///
    /// Decoder input:
    ///
    /// [batch, decoder_sequence, embedding]
    ///
    /// Encoder input:
    ///
    /// [batch, encoder_sequence, embedding]
    ///
    /// Output:
    ///
    /// [batch, decoder_sequence, embedding]
    pub fn forward(
        &self,
        decoder_input: Tensor<B, 3>,
        encoder_input: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        // ---------------------------------------------------------
        // 1. Project decoder and encoder representations
        //    into Q, K and V.
        // ---------------------------------------------------------

        let (q, k, v) =
            self.project_qkv(
                decoder_input,
                encoder_input,
            );

        // ---------------------------------------------------------
        // 2. Split Q, K and V into attention heads.
        // ---------------------------------------------------------

        let q =
            self.split_heads(q);

        let k =
            self.split_heads(k);

        let v =
            self.split_heads(v);

        // ---------------------------------------------------------
        // 3. Calculate cross-attention.
        // ---------------------------------------------------------

        let attention_output =
            self.attention(
                q,
                k,
                v,
            );

        // ---------------------------------------------------------
        // 4. Combine the attention heads.
        // ---------------------------------------------------------

        let combined =
            self.combine_heads(
                attention_output,
            );

        // ---------------------------------------------------------
        // 5. Final output projection.
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
