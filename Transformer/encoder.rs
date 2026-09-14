use burn::module::Module;
use burn::nn::{LayerNorm, LayerNormConfig};
use burn::prelude::*;

use crate::Attention::MHSA::MHSA;
use crate::FFN::FFN;

#[derive(Module, Debug)]
pub struct EncoderBlock<B: Backend> {
pub norm1: LayerNorm<B>,
pub attention: MHSA<B>,
pub norm2: LayerNorm<B>,
pub ffn: FFN<B>,
}

impl<B: Backend> EncoderBlock<B> {
pub fn new(
embedding_dimension: usize,
num_heads: usize,
ffn_hidden_dimension: usize,
device: &B::Device,
) -> Self {
assert!(embedding_dimension > 0);
assert!(num_heads > 0);
assert!(ffn_hidden_dimension > 0);
assert!(
embedding_dimension % num_heads == 0,
"Embedding dimension must be divisible by the number of attention heads"
);

    let norm1 = LayerNormConfig::new(embedding_dimension).init(device);

    let attention = MHSA::new(
        embedding_dimension,
        num_heads,
        device,
    );

    let norm2 = LayerNormConfig::new(embedding_dimension).init(device);

    let ffn = FFN::new(
        embedding_dimension,
        ffn_hidden_dimension,
        embedding_dimension,
        device,
    );

    Self {
        norm1,
        attention,
        norm2,
        ffn,
    }
}

pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
    let normalized_input = self.norm1.forward(input.clone());
    let attention_output = self.attention.forward(normalized_input);
    let x = input + attention_output;

    let normalized_x = self.norm2.forward(x.clone());
    let ffn_output = self.ffn.forward(normalized_x);

    x + ffn_output
}

pub fn embedding_dimension(&self) -> usize {
    self.attention.embedding_dimension()
}

pub fn num_heads(&self) -> usize {
    self.attention.num_heads()
}

}
