use burn::module::Module;
use burn::nn::{RotaryEncoding, RotaryEncodingConfig};
use burn::prelude::*;

#[derive(Module, Debug)]
pub struct PositionalEncoding<B: Backend> {
rotary_encoding: RotaryEncoding<B>,
dimension: usize,
max_sequence_length: usize,
theta: f32,
}

impl<B: Backend> PositionalEncoding<B> {
pub fn new(
dimension: usize,
max_sequence_length: usize,
theta: f32,
device: &B::Device,
) -> Self {
assert!(
dimension > 0,
"RoPE dimension must be greater than zero"
);

    assert!(
        dimension % 2 == 0,
        "RoPE dimension must be even"
    );

    assert!(
        max_sequence_length > 0,
        "Maximum sequence length must be greater than zero"
    );

    assert!(
        theta > 0.0,
        "RoPE theta must be greater than zero"
    );

    let rotary_encoding = RotaryEncodingConfig::new(
        max_sequence_length,
        dimension,
    )
    .with_theta(theta)
    .init(device);

    Self {
        rotary_encoding,
        dimension,
        max_sequence_length,
        theta,
    }
}

pub fn forward<const D: usize>(
    &self,
    input: Tensor<B, D>,
) -> Tensor<B, D> {
    let shape = input.shape();

    assert!(
        D >= 2,
        "RoPE input must have at least two dimensions"
    );

    let dims = shape.dims::<D>();
    let sequence_length = dims[D - 2];
    let dimension = dims[D - 1];

    assert_eq!(
        dimension,
        self.dimension,
        "RoPE input dimension ({}) does not match configured dimension ({})",
        dimension,
        self.dimension
    );

    assert!(
        sequence_length <= self.max_sequence_length,
        "Sequence length ({}) exceeds configured RoPE maximum ({})",
        sequence_length,
        self.max_sequence_length
    );

    self.rotary_encoding.forward(input)
}

pub fn dimension(&self) -> usize {
    self.dimension
}

pub fn max_sequence_length(&self) -> usize {
    self.max_sequence_length
}

pub fn theta(&self) -> f32 {
    self.theta
}

}
