mod input_module;

use input_module::{Input, Labels, normalize_input, minmax_normalize};

fn relu(x: f32) -> f32 {
    x.max(0.0)
}


fn apply_relu(values: &[f32]) -> Vec<f32> {
    values.iter().map(|&x| relu(x)).collect()
}


fn gelu(x: f32) -> f32 {
    let sqrt_2_over_pi = (2.0 / std::f32::consts::PI).sqrt();

    0.5 * x
        * (1.0
            + (sqrt_2_over_pi * (x + 0.044715 * x.powi(3))).tanh())
}

fn apply_gelu(values: &[f32]) -> Vec<f32> {
    values.iter()
        .map(|&x| gelu(x))
        .collect()
}


fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);

    let exp_values: Vec<f32> = logits
        .iter()
        .map(|&x| (x - max).exp())
        .collect();

    let sum: f32 = exp_values.iter().sum();

    exp_values
        .iter()
        .map(|&x| x / sum)
        .collect()
}


fn matrix_mul(a: &[Vec<f32>], b: &[Vec<f32>]) -> Vec<Vec<f32>> {
    assert!(!a.is_empty(), "Matrix A cannot be empty");
    assert!(!b.is_empty(), "Matrix B cannot be empty");

    let cols_a = a[0].len();
    assert!(cols_a > 0, "Matrix A cannot have zero columns");

    for row in a {
        assert_eq!(row.len(), cols_a, "Matrix A has inconsistent row lengths");
    }

    let cols_b = b[0].len();
    assert!(cols_b > 0, "Matrix B cannot have zero columns");

    for row in b {
        assert_eq!(row.len(), cols_b, "Matrix B has inconsistent row lengths");
    }

    assert_eq!(
        cols_a,
        b.len(),
        "Cannot multiply matrices: A columns ({}) != B rows ({})",
        cols_a,
        b.len()
    );

    let rows_a = a.len();
    let mut result = vec![vec![0.0; cols_b]; rows_a];

    for i in 0..rows_a {
        for j in 0..cols_b {
            for k in 0..cols_a {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    result
}

fn forward(input: &Input, weights: &[Vec<f32>], biases: &[f32]) -> Vec<f32> {
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

fn mse_loss(y: &Labels, y_pred: &[f32]) -> f32 {
    let y_vec = y.to_vec();

    assert_eq!(
        y_vec.len(),
        y_pred.len(),
        "Labels and predictions must have the same length"
    );

    y_vec
        .iter()
        .zip(y_pred.iter())
        .map(|(&target, &prediction)| (prediction - target).powi(2))
        .sum::<f32>()
        / y_vec.len() as f32
}

fn cross_entropy_loss(y: &Labels, y_pred: &[f32]) -> f32 {
    let y_vec: Vec<f32> = y.to_vec();

    assert_eq!(
        y_vec.len(),
        y_pred.len(),
        "Labels and predictions must have the same length"
    );

    let epsilon = 1e-7;
    
    y_vec
        .iter()
        .zip(y_pred.iter())
        .map(|(&target, &prediction)| {
            let pred = prediction.clamp(epsilon, 1.0 - epsilon);
            -(target * pred.ln() + (1.0 - target) * (1.0 - pred).ln())
        })
        .sum::<f32>() / y_vec.len() as f32
}

mod gradient {
    use crate::input_module::Labels;
    use crate::softmax;

    pub fn loss_gradient(y: &Labels, y_pred: &[f32]) -> Vec<f32> {
        let y_vec = y.to_vec();

        assert_eq!(
            y_vec.len(),
            y_pred.len(),
            "Labels and predictions must have the same length"
        );

        let n = y_vec.len() as f32;

        y_vec
            .iter()
            .zip(y_pred.iter())
            .map(|(&target, &prediction)| 2.0 * (prediction - target) / n)
            .collect()
    }

    pub fn softmax_cross_entropy_grad(
        probabilities: &[f32],
        target_index: usize,
    ) -> Vec<f32> {
        assert!(
            target_index < probabilities.len(),
            "Target index is outside vocabulary"
        );

        let mut gradient = probabilities.to_vec();

        gradient[target_index] -= 1.0;

        gradient
    }

    pub fn relu_gradient(x: f32) -> f32 {
        if x > 0.0 { 1.0 } else { 0.0 }
    }

    pub fn gelu_gradient(x: f32) -> f32 {
        let c = (2.0 / std::f32::consts::PI).sqrt();
        let k = 0.044715;

        let inner = c * (x + k * x.powi(3));
        let tanh_inner = inner.tanh();

        let sech2 = 1.0 - tanh_inner.powi(2);

        0.5 * (1.0 + tanh_inner)
            + 0.5
                * x
                * sech2
                * c
                * (1.0 + 3.0 * k * x.powi(2))
    }


    pub fn softmax_gradient(logits: &[f32]) -> Vec<Vec<f32>> {
    let probabilities = softmax(logits);
    let n = probabilities.len();

    let mut jacobian = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..n {
            if i == j {
                jacobian[i][j] =
                    probabilities[i] * (1.0 - probabilities[j]);
            } else {
                jacobian[i][j] =
                    -probabilities[i] * probabilities[j];
            }
        }
    }

    jacobian
}
}



struct Gradients {
    d_weights: Vec<Vec<f32>>,
    d_biases: Vec<f32>,
}

fn backward(input: &Input, weights: &[Vec<f32>], biases: &[f32], labels: &Labels) -> Gradients {
    let x = input.to_vec();

    let linear_output = matrix_mul(&input.to_matrix(), weights);
    let pre_activation: Vec<f32> = linear_output[0]
        .iter()
        .zip(biases.iter())
        .map(|(&value, &bias)| value + bias)
        .collect();

    let predictions = apply_relu(&pre_activation);

    let d_loss = gradient::loss_gradient(labels, &predictions);

    let d_z: Vec<f32> = d_loss
        .iter()
        .zip(pre_activation.iter())
        .map(|(&dl, &z)| dl * gradient::relu_gradient(z))
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

fn print_matrix(name: &str, matrix: &[Vec<f32>]) {
    println!("{}", name);
    for row in matrix {
        println!("    {:?}", row);
    }
}

fn main() {
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