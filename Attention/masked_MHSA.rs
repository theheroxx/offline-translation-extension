use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::softmax;


#[derive(Module, Debug)]
pub struct MaskedMHSA<B: Backend> {
    /// Query projection.
    pub w_q: Linear<B>,

    /// Key projection.
    pub w_k: Linear<B>,

    /// Value projection.
    pub w_v: Linear<B>,

    /// Output projection.
    pub w_o: Linear<B>,

    /// Total Transformer embedding dimension.
    pub embedding_dimension: usize,

    /// Number of attention heads.
    pub num_heads: usize,
}

impl<B: Backend> MaskedMHSA<B> {

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
        // Q projection
        // ---------------------------------------------------------

        let w_q = LinearConfig::new(
            embedding_dimension,
            embedding_dimension,
        )
        .init(device);

        // ---------------------------------------------------------
        // K projection
        // ---------------------------------------------------------

        let w_k = LinearConfig::new(
            embedding_dimension,
            embedding_dimension,
        )
        .init(device);

        // ---------------------------------------------------------
        // V projection
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

    /// Projects the input into Q, K and V.
    ///
    /// Input:
    ///
    /// [batch, sequence, embedding]
    ///
    /// Output:
    ///
    /// Q: [batch, sequence, embedding]
    /// K: [batch, sequence, embedding]
    /// V: [batch, sequence, embedding]
    ///
    /// Because this is self-attention, the same input
    /// sequence is used to create Q, K and V.
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

    /// Creates the causal attention mask.
    ///
    /// The mask has shape:
    ///
    /// [1, 1, sequence_length, sequence_length]
    ///
    /// The first two dimensions are `1` so that Burn
    /// can broadcast the mask across:
    ///
    /// batch
    /// heads
    ///
    /// Allowed:
    ///
    /// position i -> position j when j <= i
    ///
    /// Blocked:
    ///
    /// position i -> position j when j > i
    ///
    /// Example for sequence length 4:
    ///
    ///        key position
    ///          0      1      2      3
    ///
    ///  0      0    -inf   -inf   -inf
    ///  1      0      0    -inf   -inf
    ///  2      0      0      0    -inf
    ///  3      0      0      0      0
    ///
    /// A large negative value is used instead of negative
    /// infinity so that softmax effectively gives the
    /// masked positions probability zero.
    fn causal_mask(
        &self,
        sequence_length: usize,
        device: &B::Device,
    ) -> Tensor<B, 4> {
        let mut mask_data =
            vec![0.0_f32; sequence_length * sequence_length];

        for i in 0..sequence_length {
            for j in 0..sequence_length {
                let index =
                    i * sequence_length + j;

                if j > i {
                    mask_data[index] = -1.0e9;
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

    /// Performs masked scaled dot-product attention.
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
    /// Returns:
    ///
    /// [batch, heads, sequence, head_dimension]
    fn attention(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        device: &B::Device,
    ) -> Tensor<B, 4> {
        let head_dimension =
            self.embedding_dimension / self.num_heads;

        // ---------------------------------------------------------
        // 1. Transpose K
        // ---------------------------------------------------------
        //
        // K:
        //
        // [B, H, S, D]
        //
        // becomes:
        //
        // [B, H, D, S]

        let k_transposed =
            k.swap_dims(2, 3);

        // ---------------------------------------------------------
        // 2. Calculate QK^T
        // ---------------------------------------------------------
        //
        // Q:
        //
        // [B, H, S, D]
        //
        // K^T:
        //
        // [B, H, D, S]
        //
        // Result:
        //
        // [B, H, S, S]

        let scores =
            q.matmul(k_transposed);

        // ---------------------------------------------------------
        // 3. Scale by sqrt(head_dimension)
        // ---------------------------------------------------------

        let scale =
            (head_dimension as f32).sqrt();

        let scores =
            scores / scale;

        // ---------------------------------------------------------
        // 4. Create causal mask
        // ---------------------------------------------------------

        let sequence_length =
            scores.dims()[2];

        let mask =
            self.causal_mask(
                sequence_length,
                device,
            );

        // ---------------------------------------------------------
        // 5. Apply causal mask
        // ---------------------------------------------------------
        //
        // Future positions receive a very large negative
        // score.
        //
        // After softmax:
        //
        // exp(-1e9) ≈ 0
        //
        // therefore future positions receive approximately
        // zero attention probability.

        let scores =
            scores + mask;

        // ---------------------------------------------------------
        // 6. Softmax
        // ---------------------------------------------------------
        //
        // Softmax is applied over the key sequence dimension.
        //
        // [B, H, S, S]

        let attention_weights =
            softmax(scores, 3);

        // ---------------------------------------------------------
        // 7. Weighted sum of V
        // ---------------------------------------------------------
        //
        // Attention weights:
        //
        // [B, H, S, S]
        //
        // V:
        //
        // [B, H, S, D]
        //
        // Result:
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
        // Keep the device before consuming `input`.
        let device = input.device();

        // ---------------------------------------------------------
        // 1. Project input into Q, K and V
        // ---------------------------------------------------------

        let (q, k, v) =
            self.project_qkv(input);

        // ---------------------------------------------------------
        // 2. Split Q, K and V into multiple heads
        // ---------------------------------------------------------

        let q =
            self.split_heads(q);

        let k =
            self.split_heads(k);

        let v =
            self.split_heads(v);

        // ---------------------------------------------------------
        // 3. Masked scaled dot-product attention
        // ---------------------------------------------------------

        let attention_output =
            self.attention(
                q,
                k,
                v,
                &device,
            );

        // ---------------------------------------------------------
        // 4. Combine all attention heads
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
