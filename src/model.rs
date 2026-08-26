use crate::input_module::Input;
use crate::input_module::Labels;
use crate::matrix::matrix_mul;
use crate::activations::apply_relu;

use crate::gradients;

pub struct Gradients {
    pub d_weights: Vec<Vec<f32>>,
    pub d_biases: Vec<f32>,
}

pub fn forward(input: &Input, weights: &[Vec<f32>], biases: &[f32]) -> Vec<f32> {
    let x = input.to_matrix();
    let linear_output = matrix_mul(&x, weights);
    let values = &linear_output[0];

    assert_eq!(
        values.len(),
        biases.len(),
        "Bias size ({}) must match number of outputs ({})",
        biases.len(),
        values.len()
    );

    let pre_activation: Vec<f32> = values
        .iter()
        .zip(biases.iter())
        .map(|(&value, &bias)| value + bias)
        .collect();

    apply_relu(&pre_activation)
}


pub fn backward(input: &Input, weights: &[Vec<f32>], biases: &[f32], labels: &Labels) -> Gradients {
    let x = input.to_vec();

    let linear_output = matrix_mul(&input.to_matrix(), weights);
    let pre_activation: Vec<f32> = linear_output[0]
        .iter()
        .zip(biases.iter())
        .map(|(&value, &bias)| value + bias)
        .collect();

    let predictions = apply_relu(&pre_activation);

    let d_loss = gradients::loss_gradient(labels, &predictions);

    let d_z: Vec<f32> = d_loss
        .iter()
        .zip(pre_activation.iter())
        .map(|(&dl, &z)| dl * gradients::relu_gradient(z))
        .collect();

    let mut d_weights = vec![vec![0.0; weights[0].len()]; weights.len()];

    for i in 0..weights.len() {
        for j in 0..weights[0].len() {
            d_weights[i][j] = x[i] * d_z[j];
        }
    }

    let d_biases = d_z.clone();

    Gradients {
        d_weights,
        d_biases,
    }
}
