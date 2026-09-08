use burn::prelude::*;


// ============================================================
// INPUT
// ============================================================
//
// Represents one input sample containing four features.
//
// This is a data container, not a neural-network layer.
//
// For the Transformer itself, token IDs will be used instead.
//


#[derive(Debug, Clone, Copy)]
pub struct Input {
    pub x1: f32,
    pub x2: f32,
    pub x3: f32,
    pub x4: f32,
}

impl Input {

    // --------------------------------------------------------
    // Convert to Rust vector
    // --------------------------------------------------------

    pub fn to_vec(&self) -> Vec<f32> {
        vec![
            self.x1,
            self.x2,
            self.x3,
            self.x4,
        ]
    }


    // --------------------------------------------------------
    // Convert to matrix
    // --------------------------------------------------------
    //
    // Shape:
    //
    //     [1, 4]
    //
    // --------------------------------------------------------

    pub fn to_matrix(&self) -> Vec<Vec<f32>> {
        vec![self.to_vec()]
    }


    // --------------------------------------------------------
    // Convert directly to Burn Tensor
    // --------------------------------------------------------
    //
    // Shape:
    //
    //     [1, 4]
    //
    // --------------------------------------------------------

    pub fn to_tensor<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Tensor<B, 2> {
        Tensor::<B, 1>::from_floats(
            self.to_vec().as_slice(),
            device,
        )
        .reshape([1, 4])
    }
}


// ============================================================
// LABELS
// ============================================================

#[derive(Debug, Clone, Copy)]
pub struct Labels {
    pub x1: f32,
    pub x2: f32,
    pub x3: f32,
    pub x4: f32,
}

impl Labels {

    // --------------------------------------------------------
    // Convert to Rust vector
    // --------------------------------------------------------

    pub fn to_vec(&self) -> Vec<f32> {
        vec![
            self.x1,
            self.x2,
            self.x3,
            self.x4,
        ]
    }


    // --------------------------------------------------------
    // Convert directly to Burn Tensor
    // --------------------------------------------------------
    //
    // Shape:
    //
    //     [1, 4]
    //
    // --------------------------------------------------------

    pub fn to_tensor<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Tensor<B, 2> {
        Tensor::<B, 1>::from_floats(
            self.to_vec().as_slice(),
            device,
        )
        .reshape([1, 4])
    }
}


// ============================================================
// STANDARD NORMALIZATION
// ============================================================
//
// Normalization:
//
//     x' = (x - mean) / std
//
// This operates on one Input sample.
//

pub fn normalize_input(input: &Input) -> Input {

    let values = input.to_vec();

    let n = values.len() as f32;

    let mean =
        values.iter().sum::<f32>() / n;

    let variance =
        values
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>()
            / n;

    let std =
        variance.sqrt() + 1e-8;

    Input {
        x1: (input.x1 - mean) / std,
        x2: (input.x2 - mean) / std,
        x3: (input.x3 - mean) / std,
        x4: (input.x4 - mean) / std,
    }
}


// ============================================================
// MIN-MAX NORMALIZATION
// ============================================================
//
// Normalization:
//
//     x' = (x - min) / (max - min)
//
// Handles the constant-input case where:
//
//     max == min
//
// In that situation all values are returned as zero.
//

pub fn minmax_normalize(input: &Input) -> Input {

    let values = input.to_vec();

    let min_val =
        values
            .iter()
            .fold(
                f32::INFINITY,
                |a, &b| a.min(b),
            );

    let max_val =
        values
            .iter()
            .fold(
                f32::NEG_INFINITY,
                |a, &b| a.max(b),
            );

    let range =
        max_val - min_val;

    if range.abs() < 1e-8 {
        return Input {
            x1: 0.0,
            x2: 0.0,
            x3: 0.0,
            x4: 0.0,
        };
    }

    Input {
        x1: (input.x1 - min_val) / range,
        x2: (input.x2 - min_val) / range,
        x3: (input.x3 - min_val) / range,
        x4: (input.x4 - min_val) / range,
    }
}
