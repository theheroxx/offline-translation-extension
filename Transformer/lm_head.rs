
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;

#[derive(Module, Debug)]
pub struct LMHead<B: Backend> {
    pub linear: Linear<B>,
    pub embedding_dimension: usize,
    pub target_vocab_size: usize,
}

impl<B: Backend> LMHead<B> {
    pub fn new(
        embedding_dimension: usize,
        target_vocab_size: usize,
        device: &B::Device,
    ) -> Self {
        assert!(
            embedding_dimension > 0,
            "Embedding dimension must be greater than zero"
        );

        assert!(
            target_vocab_size > 0,
            "Target vocabulary size must be greater than zero"
        );

        let linear =
            LinearConfig::new(embedding_dimension, target_vocab_size)
                .init(device);

        Self {
            linear,
            embedding_dimension,
            target_vocab_size,
        }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_, _, embedding_dimension] = input.shape().dims::<3>();

        assert_eq!(
            embedding_dimension,
            self.embedding_dimension,
            "LM Head input dimension does not match embedding dimension"
        );

        self.linear.forward(input)
    }

    pub fn embedding_dimension(&self) -> usize {
        self.embedding_dimension
    }

    pub fn target_vocab_size(&self) -> usize {
        self.target_vocab_size
    }
}
