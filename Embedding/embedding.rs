#[derive(Debug, Clone)]
pub struct Embedding {
    /// Embedding matrix:
    ///
    /// rows    = vocabulary size
    /// columns = embedding dimension
    ///
    /// Shape:
    /// [vocab_size][embedding_dim]
    pub weights: Vec<Vec<f32>>,

    pub vocab_size: usize,
    pub embedding_dim: usize,
}

impl Embedding {

    // CREATE EMBEDDING


    pub fn new(
        vocab_size: usize,
        embedding_dim: usize,
    ) -> Self {
        assert!(
            vocab_size > 0,
            "Vocabulary size must be greater than zero"
        );

        assert!(
            embedding_dim > 0,
            "Embedding dimension must be greater than zero"
        );

        // Xavier-style initialization.
        //
        // limit = sqrt(6 / (vocab_size + embedding_dim))
        //
        // For now we use a deterministic initialization so that
        // your results are reproducible while learning.

        let limit =
            (6.0 / (vocab_size + embedding_dim) as f32).sqrt();

        let mut weights =
            vec![vec![0.0; embedding_dim]; vocab_size];

        for i in 0..vocab_size {
            for j in 0..embedding_dim {
                // Simple deterministic pseudo-random value.
                let value =
                    ((i * 31 + j * 17) % 1000) as f32 / 1000.0;

                weights[i][j] =
                    (value * 2.0 - 1.0) * limit;
            }
        }

        Self {
            weights,
            vocab_size,
            embedding_dim,
        }
    }



    // FORWARD

    //
    // Input:
    //
    //     token IDs
    //
    //     [2, 17, 45, 8, 3]
    //
    // Output:
    //
    //     [5][embedding_dim]
    //


    pub fn forward(
        &self,
        token_ids: &[usize],
    ) -> Vec<Vec<f32>> {
        let mut output =
            Vec::with_capacity(token_ids.len());

        for &token_id in token_ids {
            assert!(
                token_id < self.vocab_size,
                "Token ID {} is outside vocabulary size {}",
                token_id,
                self.vocab_size
            );

            output.push(
                self.weights[token_id].clone()
            );
        }

        output
    }



    // BACKWARD

    //
    // Embedding is different from a normal Linear layer.
    //
    // We don't calculate:
    //
    //     X^T * gradient
    //
    // for every vocabulary row.
    //
    // Only the embedding vectors that were actually used
    // receive gradients.
    //


    pub fn backward(
        &self,
        token_ids: &[usize],
        output_gradients: &[Vec<f32>],
    ) -> Vec<Vec<f32>> {
        assert_eq!(
            token_ids.len(),
            output_gradients.len(),
            "Token IDs and output gradients must have the same sequence length"
        );

        let mut gradients =
            vec![
                vec![0.0; self.embedding_dim];
                self.vocab_size
            ];

        for (position, &token_id) in token_ids.iter().enumerate() {
            assert!(
                token_id < self.vocab_size,
                "Token ID {} is outside vocabulary size {}",
                token_id,
                self.vocab_size
            );

            assert_eq!(
                output_gradients[position].len(),
                self.embedding_dim,
                "Gradient dimension does not match embedding dimension"
            );

            for d in 0..self.embedding_dim {
                gradients[token_id][d] +=
                    output_gradients[position][d];
            }
        }

        gradients
    }



    // UPDATE


    pub fn update(
        &mut self,
        gradients: &[Vec<f32>],
        learning_rate: f32,
    ) {
        assert_eq!(
            gradients.len(),
            self.vocab_size,
            "Gradient vocabulary size does not match embedding vocabulary size"
        );

        for token_id in 0..self.vocab_size {
            assert_eq!(
                gradients[token_id].len(),
                self.embedding_dim,
                "Gradient dimension does not match embedding dimension"
            );

            for d in 0..self.embedding_dim {
                self.weights[token_id][d] -=
                    learning_rate * gradients[token_id][d];
            }
        }
    }



    // GET ONE EMBEDDING VECTOR

    pub fn get(&self,token_id: usize,) -> &[f32] {
        assert!(
            token_id < self.vocab_size,
            "Token ID {} is outside vocabulary size {}",
            token_id,
            self.vocab_size
        );

        &self.weights[token_id]
    }



    // EMBEDDING SIZE


    pub fn vocab_size(&self) -> usize {self.vocab_size}

    pub fn embedding_dim(&self) -> usize {self.embedding_dim}



    // PRINT


    pub fn print(
        &self,
        token_ids: &[usize],
    ) {
        println!();
        println!("========================================");
        println!("EMBEDDINGS");
        println!("========================================");

        let embeddings = self.forward(token_ids);

        for (position, (&token_id, embedding)) in
            token_ids.iter().zip(embeddings.iter()).enumerate()
        {
            println!();
            println!(
                "Position {} | Token ID {}",
                position,
                token_id
            );

            println!("{:?}", embedding);
        }
    }
}