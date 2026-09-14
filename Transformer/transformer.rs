use burn::module::Module;
use burn::prelude::*;

use crate::Transformer::decoder::DecoderBlock;
use crate::Transformer::encoder::EncoderBlock;

#[derive(Module, Debug)]
pub struct Transformer<B: Backend> {
pub encoder_layers: Vec<EncoderBlock<B>>,
pub decoder_layers: Vec<DecoderBlock<B>>,
pub num_encoder_layers: usize,
pub num_decoder_layers: usize,
pub embedding_dimension: usize,
pub num_heads: usize,
pub ffn_hidden_dimension: usize,
}

impl<B: Backend> Transformer<B> {
pub fn new(
num_encoder_layers: usize,
num_decoder_layers: usize,
embedding_dimension: usize,
num_heads: usize,
ffn_hidden_dimension: usize,
device: &B::Device,
) -> Self {
assert!(
num_encoder_layers > 0,
"Number of encoder layers must be greater than zero"
);

    assert!(
        num_decoder_layers > 0,
        "Number of decoder layers must be greater than zero"
    );

    assert!(
        embedding_dimension > 0,
        "Embedding dimension must be greater than zero"
    );

    assert!(
        num_heads > 0,
        "Number of attention heads must be greater than zero"
    );

    assert!(
        ffn_hidden_dimension > 0,
        "FFN hidden dimension must be greater than zero"
    );

    assert!(
        embedding_dimension % num_heads == 0,
        "Embedding dimension must be divisible by the number of attention heads"
    );

    let encoder_layers = (0..num_encoder_layers)
        .map(|_| {
            EncoderBlock::new(
                embedding_dimension,
                num_heads,
                ffn_hidden_dimension,
                device,
            )
        })
        .collect();

    let decoder_layers = (0..num_decoder_layers)
        .map(|_| {
            DecoderBlock::new(
                embedding_dimension,
                num_heads,
                ffn_hidden_dimension,
                device,
            )
        })
        .collect();

    Self {
        encoder_layers,
        decoder_layers,
        num_encoder_layers,
        num_decoder_layers,
        embedding_dimension,
        num_heads,
        ffn_hidden_dimension,
    }
}

pub fn encode(
    &self,
    source_embeddings: Tensor<B, 3>,
) -> Tensor<B, 3> {
    let mut encoder_output = source_embeddings;

    for encoder_layer in &self.encoder_layers {
        encoder_output = encoder_layer.forward(encoder_output);
    }

    encoder_output
}

pub fn decode(
    &self,
    target_embeddings: Tensor<B, 3>,
    encoder_output: Tensor<B, 3>,
) -> Tensor<B, 3> {
    let mut decoder_output = target_embeddings;

    for decoder_layer in &self.decoder_layers {
        decoder_output = decoder_layer.forward(
            decoder_output,
            encoder_output.clone(),
        );
    }

    decoder_output
}

pub fn forward(
    &self,
    source_embeddings: Tensor<B, 3>,
    target_embeddings: Tensor<B, 3>,
) -> Tensor<B, 3> {
    let encoder_output = self.encode(source_embeddings);

    self.decode(
        target_embeddings,
        encoder_output,
    )
}

pub fn num_encoder_layers(&self) -> usize {
    self.num_encoder_layers
}

pub fn num_decoder_layers(&self) -> usize {
    self.num_decoder_layers
}

pub fn embedding_dimension(&self) -> usize {
    self.embedding_dimension
}

pub fn num_heads(&self) -> usize {
    self.num_heads
}

pub fn ffn_hidden_dimension(&self) -> usize {
    self.ffn_hidden_dimension
}

}
