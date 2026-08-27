mod FFN;
mod Tokenizer;

use Tokenizer::load_dataset;
use FFN::input_module::{Input, Labels, normalize_input, minmax_normalize};
use FFN::activations::{relu, gelu, apply_relu, apply_gelu, softmax};
use FFN::model::{forward, backward};
use FFN::losses::{mse_loss, cross_entropy_loss};
use FFN::gradients::{
    relu_gradient, gelu_gradient, loss_gradient, 
    softmax_cross_entropy_grad, softmax_gradient,
};

fn print_matrix(name: &str, matrix: &[Vec<f32>]) {
    println!("{}", name);
    for row in matrix {
        println!("    {:?}", row);

        
    }
}

fn main() {


        let dataset = load_dataset("data")
        .expect("Failed to load dataset");

    println!("Loaded {} translation pairs", dataset.len());

    for (i, pair) in dataset.iter().take(5).enumerate() {
        println!();
        println!("PAIR {}", i + 1);
        println!("SOURCE: {}", pair.source);
        println!("TARGET: {}", pair.target);
    }
    let raw_input = Input {
        x1: 11.0,
        x2: 102.0,
        x3: 440.0,
        x4: 60.0,
    };

    let input = normalize_input(&raw_input);

    let labels = Labels {
        x1: 0.0,
        x2: 1.0,
        x3: 1.0,
        x4: 0.0,
    };

    let mut weights = vec![
        vec![0.01, 0.01, 0.03, 0.91],
        vec![0.03, 0.41, 0.11, 0.01],
        vec![0.01, 0.07, 0.01, 0.01],
        vec![0.02, 0.102, 0.01, 0.01],
    ];

    let mut biases = vec![1.0, 1.0, 1.0, 1.0];

    let learning_rate = 0.01;
    let epochs = 100;

    println!("========================================");
    println!("INITIAL STATE");
    println!("========================================");

    println!("Raw Input: {:?}", raw_input.to_vec());
    println!("Normalized Input: {:?}", input.to_vec());
    println!("Labels: {:?}", labels.to_vec());
    print_matrix("Weights:", &weights);
    println!("Biases: {:?}", biases);

    let initial_predictions = forward(&input, &weights, &biases);
    let initial_loss = mse_loss(&labels, &initial_predictions);

    println!("Predictions: {:?}", initial_predictions);
    println!("Initial Loss: {}", initial_loss);

    for epoch in 0..epochs {
        println!();
        println!("========================================");
        println!("EPOCH {}", epoch);
        println!("========================================");

        let predictions = forward(&input, &weights, &biases);
        let current_loss = mse_loss(&labels, &predictions);

        println!("Predictions BEFORE update: {:?}", predictions);
        println!("Loss BEFORE update: {}", current_loss);

        let gradients = backward(&input, &weights, &biases, &labels);

        println!("d_biases: {:?}", gradients.d_biases);
        print_matrix("d_weights:", &gradients.d_weights);

        let old_weights = weights.clone();
        let old_biases = biases.clone();

        for i in 0..weights.len() {
            for j in 0..weights[0].len() {
                weights[i][j] -= learning_rate * gradients.d_weights[i][j];
            }
        }

        for i in 0..biases.len() {
            biases[i] -= learning_rate * gradients.d_biases[i];
        }

        println!();
        println!("PARAMETER UPDATE");
        print_matrix("Weights BEFORE:", &old_weights);
        print_matrix("Weights AFTER:", &weights);
        println!("Biases BEFORE: {:?}", old_biases);
        println!("Biases AFTER: {:?}", biases);

        let new_predictions = forward(&input, &weights, &biases);
        let new_loss = mse_loss(&labels, &new_predictions);

        println!();
        println!("Predictions AFTER update: {:?}", new_predictions);
        println!("Loss AFTER update: {}", new_loss);
    }

    println!();
    println!("========================================");
    println!("FINAL STATE");
    println!("========================================");

    let final_predictions = forward(&input, &weights, &biases);
    let final_loss = mse_loss(&labels, &final_predictions);

    println!("Final predictions: {:?}", final_predictions);
    println!("Final loss: {}", final_loss);
    print_matrix("Final weights:", &weights);
    println!("Final biases: {:?}", biases);
}