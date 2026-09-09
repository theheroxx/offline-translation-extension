use burn::prelude::*;
use burn::tensor::activation::log_softmax;


pub fn mse_loss<B: Backend>(
    predictions: Tensor<B, 2>,
    targets: Tensor<B, 2>,
) -> Tensor<B, 1> {
    let difference = predictions - targets;

    difference
        .powf_scalar(2.0)
        .mean()
}



pub fn cross_entropy_loss<B: Backend>(
    logits: Tensor<B, 3>,
    targets: Tensor<B, 2, Int>,
) -> Tensor<B, 1> {

    let log_probabilities =
        log_softmax(logits, 2);


    let targets =
        targets.unsqueeze_dim(2);



let target_log_probabilities =
    log_probabilities.gather(2, targets);

let target_log_probabilities: Tensor<B, 2> =
    target_log_probabilities.squeeze_dims(&[2]);

target_log_probabilities.neg().mean()
}