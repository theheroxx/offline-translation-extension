use burn::module::Module;
use burn::nn::{Embedding as BurnEmbedding, EmbeddingConfig};
use burn::prelude::*;


#[derive(Module, Debug)]
pub struct Embedding<B: Backend> {
    embedding: BurnEmbedding<B>,
    vocab_size: usize,
    embedding_dim: usize,
}

impl<B: Backend> Embedding<B> {

    // ============================================================
    // CREATE EMBEDDING
    // ============================================================

    pub fn new(
        vocab_size: usize,
        embedding_dim: usize,
        device: &B::Device,
    ) -> Self {
        assert!(
            vocab_size > 0,
            "Vocabulary size must be greater than zero"
        );

        assert!(
            embedding_dim > 0,
            "Embedding dimension must be greater than zero"
        );

        let embedding =
            EmbeddingConfig::new(vocab_size, embedding_dim)
                .init(device);

        Self {
            embedding,
            vocab_size,
            embedding_dim,
        }
    }


    // ============================================================
    // FORWARD
    // ============================================================

    //
    // Input:
    //
    //     token IDs
    //
    //     [batch, sequence]
    //
    // Example:
    //
    //     [
    //         [2, 17, 45, 8, 3]
    //     ]
    //
    // Output:
    //
    //     [batch, sequence, embedding_dim]
    //
    // ============================================================

    pub fn forward(
        &self,
        token_ids: Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        self.embedding.forward(token_ids)
    }


    // ============================================================
    // FORWARD FROM A SINGLE TOKEN SEQUENCE
    // ============================================================

    //
    // Convenience function for:
    //
    //     [sequence]
    //
    // →  [1, sequence]
    //
    // →  [1, sequence, embedding_dim]
    //
    // This is useful while testing the tokenizer before
    // introducing a real batch dimension.
    //
    // ============================================================

    pub fn forward_sequence(
        &self,
        token_ids: &[usize],
        device: &B::Device,
    ) -> Tensor<B, 3> {
        assert!(
            !token_ids.is_empty(),
            "Token sequence must not be empty"
        );

        for &token_id in token_ids {
            assert!(
                token_id < self.vocab_size,
                "Token ID {} is outside vocabulary size {}",
                token_id,
                self.vocab_size
            );
        }

        let token_ids_i64: Vec<i64> = token_ids
            .iter()
            .map(|&id| id as i64)
            .collect();

        let input = Tensor::<B, 1, Int>::from_ints(
            token_ids_i64.as_slice(),
            device,
        )
        .reshape([1, token_ids.len()]);

        self.forward(input)
    }


    // ============================================================
    // GET ONE EMBEDDING VECTOR
    // ============================================================

    //
    // Returns one embedding vector as:
    //
    //     [1, 1, embedding_dim]
    //
    // This keeps the operation inside Burn's tensor system.
    //
    // ============================================================

    pub fn get(
        &self,
        token_id: usize,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        assert!(
            token_id < self.vocab_size,
            "Token ID {} is outside vocabulary size {}",
            token_id,
            self.vocab_size
        );

        let token = Tensor::<B, 1, Int>::from_ints(
            [token_id as i64],
            device,
        )
        .reshape([1, 1]);

        self.forward(token)
    }


    // ============================================================
    // VOCABULARY SIZE
    // ============================================================

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }


    // ============================================================
    // EMBEDDING SIZE
    // ============================================================

    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
}
