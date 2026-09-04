use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::relu;


#[derive(Module, Debug)]
pub struct FFN<B: Backend> {
    pub linear1: Linear<B>,
    pub linear2: Linear<B>,
}

impl<B: Backend> FFN<B> {
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        output_size: usize,
        device: &B::Device,
    ) -> Self {
        let linear1 = LinearConfig::new(input_size, hidden_size)
            .init(device);

        let linear2 = LinearConfig::new(hidden_size, output_size)
            .init(device);

        Self {
            linear1,
            linear2,
        }
    }


    pub fn forward(
        &self,
        input: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let hidden = self.linear1.forward(input);

        let hidden = relu(hidden);

        self.linear2.forward(hidden)
    }
}