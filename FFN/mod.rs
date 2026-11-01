// FFN/mod.rs

pub mod activations;
pub mod gradients;
pub mod input_module;
pub mod losses;
pub mod matrix;
pub mod model;

pub use model::FFN;

pub use input_module::{Input, Labels, minmax_normalize, normalize_input};

pub use activations::{apply_relu, gelu, relu};

pub use losses::{cross_entropy_loss, mse_loss};

pub use matrix::matrix_mul;
