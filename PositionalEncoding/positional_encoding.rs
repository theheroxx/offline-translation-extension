use std::f32::consts::PI;

/// Rotary + Polar positional encoding.
///
/// The idea is:
///
/// 1. Split the embedding into pairs.
/// 2. Treat every pair as a 2D coordinate (x, y).
/// 3. Convert it conceptually to polar coordinates:
///
///        r     = sqrt(x² + y²)
///        theta = atan2(y, x)
///
/// 4. Add a position-dependent rotation to theta.
/// 5. Convert back to Cartesian coordinates.
///
/// This is the core idea behind RoPE:
///
///     x' = x cos(mθ) - y sin(mθ)
///     y' = x sin(mθ) + y cos(mθ)
///
/// where `m` is the token position.
///
/// Unlike vanilla sinusoidal positional encoding,
/// the positional information is applied through rotation.
pub struct PositionalEncoding {
    pub dimension: usize,
    pub max_sequence_length: usize,
    pub theta: f32,
}

impl PositionalEncoding {
    /// Create a positional encoding module.
    ///
    /// `dimension` must be even because RoPE operates
    /// on pairs of dimensions.
    pub fn new(
        dimension: usize,
        max_sequence_length: usize,
        theta: f32,
    ) -> Self {
        assert!(
            dimension > 0,
            "Embedding dimension must be greater than zero"
        );

        assert!(
            dimension % 2 == 0,
            "Embedding dimension must be even for RoPE"
        );

        assert!(
            max_sequence_length > 0,
            "Maximum sequence length must be greater than zero"
        );

        assert!(
            theta > 0.0,
            "RoPE theta must be greater than zero"
        );

        Self {
            dimension,
            max_sequence_length,
            theta,
        }
    }

    /// Standard Transformer RoPE frequency.
    ///
    /// For pair index `i`:
    ///
    ///     frequency = 1 / theta^(2i / dimension)
    ///
    fn frequency(&self, pair_index: usize) -> f32 {
        let exponent =
            (2.0 * pair_index as f32) / self.dimension as f32;

        1.0 / self.theta.powf(exponent)
    }

    /// Calculate the rotation angle for a position and pair.
    fn rotation_angle(
        &self,
        position: usize,
        pair_index: usize,
    ) -> f32 {
        position as f32 * self.frequency(pair_index)
    }

    /// Apply RoPE to one embedding vector.
    ///
    /// Input:
    ///
    ///     [x0, x1, x2, x3, ...]
    ///
    /// Dimensions are processed as:
    ///
    ///     (x0, x1)
    ///     (x2, x3)
    ///     ...
    ///
    /// Each pair is rotated according to its frequency.
    pub fn apply(
        &self,
        embedding: &[f32],
        position: usize,
    ) -> Vec<f32> {
        assert_eq!(
            embedding.len(),
            self.dimension,
            "Embedding dimension ({}) does not match positional encoding dimension ({})",
            embedding.len(),
            self.dimension
        );

        assert!(
            position < self.max_sequence_length,
            "Position {} exceeds maximum sequence length {}",
            position,
            self.max_sequence_length
        );

        let mut output = vec![0.0; self.dimension];

        for pair_index in 0..(self.dimension / 2) {
            let i = pair_index * 2;

            let x = embedding[i];
            let y = embedding[i + 1];

            let angle =
                self.rotation_angle(position, pair_index);

            let cos_theta = angle.cos();
            let sin_theta = angle.sin();

            // ------------------------------------------------
            // Polar-coordinate interpretation
            // ------------------------------------------------
            //
            // r     = sqrt(x² + y²)
            // theta = atan2(y, x)
            //
            // Then:
            //
            // theta' = theta + angle
            //
            // x' = r cos(theta')
            // y' = r sin(theta')
            //
            // We use the equivalent Cartesian rotation below.
            // ------------------------------------------------

            output[i] =
                x * cos_theta
                - y * sin_theta;

            output[i + 1] =
                x * sin_theta
                + y * cos_theta;
        }

        output
    }

    /// Apply RoPE to an entire sequence.
    ///
    /// Input:
    ///
    ///     sequence[position][embedding_dimension]
    ///
    /// Output:
    ///
    ///     rotated sequence
    pub fn apply_sequence(
        &self,
        embeddings: &[Vec<f32>],
    ) -> Vec<Vec<f32>> {
        assert!(
            embeddings.len() <= self.max_sequence_length,
            "Sequence length {} exceeds maximum {}",
            embeddings.len(),
            self.max_sequence_length
        );

        embeddings
            .iter()
            .enumerate()
            .map(|(position, embedding)| {
                self.apply(embedding, position)
            })
            .collect()
    }

    /// Apply RoPE directly to Query and Key vectors.
    ///
    /// This is the form normally used inside Transformer attention.
    pub fn apply_to_qk(
        &self,
        query: &[f32],
        key: &[f32],
        position: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        assert_eq!(
            query.len(),
            self.dimension,
            "Query dimension does not match RoPE dimension"
        );

        assert_eq!(
            key.len(),
            self.dimension,
            "Key dimension does not match RoPE dimension"
        );

        let rotated_query =
            self.apply(query, position);

        let rotated_key =
            self.apply(key, position);

        (rotated_query, rotated_key)
    }

    /// Apply RoPE to a batch of Query vectors.
    pub fn apply_to_queries(
        &self,
        queries: &[Vec<f32>],
    ) -> Vec<Vec<f32>> {
        assert!(
            queries.len() <= self.max_sequence_length,
            "Sequence length exceeds maximum sequence length"
        );

        queries
            .iter()
            .enumerate()
            .map(|(position, query)| {
                self.apply(query, position)
            })
            .collect()
    }

    /// Apply RoPE to a batch of Key vectors.
    pub fn apply_to_keys(
        &self,
        keys: &[Vec<f32>],
    ) -> Vec<Vec<f32>> {
        assert!(
            keys.len() <= self.max_sequence_length,
            "Sequence length exceeds maximum sequence length"
        );

        keys
            .iter()
            .enumerate()
            .map(|(position, key)| {
                self.apply(key, position)
            })
            .collect()
    }

    /// Calculate the polar representation of an embedding pair.
    ///
    /// Returns:
    ///
    ///     (radius, angle)
    pub fn to_polar(
        x: f32,
        y: f32,
    ) -> (f32, f32) {
        let radius = (x * x + y * y).sqrt();
        let angle = y.atan2(x);

        (radius, angle)
    }

    /// Rotate a single 2D point using polar coordinates.
    pub fn rotate_polar(
        x: f32,
        y: f32,
        rotation: f32,
    ) -> (f32, f32) {
        let (radius, angle) =
            Self::to_polar(x, y);

        let new_angle =
            angle + rotation;

        let new_x =
            radius * new_angle.cos();

        let new_y =
            radius * new_angle.sin();

        (new_x, new_y)
    }

    /// Debug information.
    pub fn print_info(&self) {
        println!("========================================");
        println!("POSITIONAL ENCODING");
        println!("========================================");

        println!(
            "Dimension: {}",
            self.dimension
        );

        println!(
            "Maximum sequence length: {}",
            self.max_sequence_length
        );

        println!(
            "RoPE theta: {}",
            self.theta
        );

        println!(
            "Number of dimension pairs: {}",
            self.dimension / 2
        );

        println!(
            "Base frequency: {}",
            self.frequency(0)
        );

        println!("========================================");
    }

    /// Print frequencies used by each dimension pair.
    pub fn print_frequencies(&self) {
        println!();
        println!("========================================");
        println!("RoPE FREQUENCIES");
        println!("========================================");

        for pair_index in 0..(self.dimension / 2) {
            let frequency =
                self.frequency(pair_index);

            println!(
                "Pair {:>4}: frequency = {:.8}",
                pair_index,
                frequency
            );
        }
    }
}