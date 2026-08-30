use crate::FFN::input_module::Labels;
use crate::FFN::activations::softmax;


// MSE GRADIENT

pub fn loss_gradient(
    y: &Labels,
    y_pred: &[f32],
) -> Vec<f32> {
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
        .map(|(&target, &prediction)| {
            2.0 * (prediction - target) / n
        })
        .collect()
}


// SOFTMAX + CROSS-ENTROPY GRADIENT
//
// For:
//
//     probabilities = softmax(logits)
//
// and:
//
//     L = -log(probability[target])
//
// the combined gradient is:
//
//     dL/dlogits = probabilities - one_hot(target)
//

pub fn softmax_cross_entropy_grad(
    probabilities: &[f32],
    target_index: usize,
) -> Vec<f32> {
    assert!(
        !probabilities.is_empty(),
        "Probabilities cannot be empty"
    );

    assert!(
        target_index < probabilities.len(),
        "Target index {} is outside vocabulary of size {}",
        target_index,
        probabilities.len()
    );

    let mut gradient = probabilities.to_vec();

    gradient[target_index] -= 1.0;

    gradient
}


// ReLU GRADIENT

pub fn relu_gradient(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else {
        0.0
    }
}


// GELU GRADIENT
//
// Derivative of the tanh-approximated GELU:
//
// GELU(x) = 0.5*x*(1 + tanh(c*(x + k*x^3)))
//
// where:
//
// c = sqrt(2/pi)
// k = 0.044715
//

pub fn gelu_gradient(x: f32) -> f32 {
    let c = (2.0 / std::f32::consts::PI).sqrt();
    let k = 0.044715;

    let inner = c * (x + k * x.powi(3));

    let tanh_inner = inner.tanh();

    let sech_squared = 1.0 - tanh_inner.powi(2);

    0.5 * (1.0 + tanh_inner)
        + 0.5
            * x
            * sech_squared
            * c
            * (1.0 + 3.0 * k * x.powi(2))
}


// SOFTMAX JACOBIAN
//
// This is the full derivative:
//
//     J_ij = p_i (delta_ij - p_j)
//
// We generally DON'T need this for a Transformer when
// softmax is immediately followed by cross-entropy.
//
// softmax_cross_entropy_grad() is much more efficient.
//

pub fn softmax_gradient(
    logits: &[f32],
) -> Vec<Vec<f32>> {
    let probabilities = softmax(logits);

    let n = probabilities.len();

    if n == 0 {
        return Vec::new();
    }

    let mut jacobian = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..n {
            if i == j {
                jacobian[i][j] =
                    probabilities[i] * (1.0 - probabilities[i]);
            } else {
                jacobian[i][j] =
                    -probabilities[i] * probabilities[j];
            }
        }
    }

    jacobian
}