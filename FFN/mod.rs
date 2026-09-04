// FFN/mod.rs

pub mod activations;
pub mod gradients;
pub mod input_module;
pub mod losses;
pub mod matrix;
pub mod model;

pub use model::FFN;

pub use input_module::{
    Input,
    Labels,
    normalize_input,
    minmax_normalize,
};

pub use activations::{
    relu,
    apply_relu,
    gelu,
};

// Re-export loss functions.
pub use losses::{
    mse_loss,
    cross_entropy_loss,
};


pub use matrix::matrix_mul;