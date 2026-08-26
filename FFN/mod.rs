// FFN/mod.rs

pub mod activations;
pub mod gradients;
pub mod input_module;
pub mod losses;
pub mod matrix;
pub mod model;
pub mod optimizer;

// Re-export commonly used items (optional)
pub use input_module::{Input, Labels, normalize_input, minmax_normalize};
pub use activations::{relu, apply_relu, gelu};
pub use losses::{mse_loss, cross_entropy_loss};
pub use matrix::matrix_mul;
pub use model::{forward, backward, Gradients};