pub fn relu(x: f32) -> f32 {
    x.max(0.0)
}

pub fn apply_relu(values: &[f32]) -> Vec<f32> {
    values.iter().map(|&x| relu(x)).collect()
}


/// GELU using the tanh approximation used commonly in Transformers.
///
/// GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) *
///                (x + 0.044715 * x^3)))
pub fn gelu(x: f32) -> f32 {
    let sqrt_2_over_pi = (2.0 / std::f32::consts::PI).sqrt();

    0.5 * x
        * (1.0
            + (sqrt_2_over_pi
                * (x + 0.044715 * x.powi(3)))
                .tanh())
}

pub fn apply_gelu(values: &[f32]) -> Vec<f32> {
    values.iter().map(|&x| gelu(x)).collect()
}


/// Derivative of the tanh-approximated GELU.
pub fn gelu_gradient(x: f32) -> f32 {
    let sqrt_2_over_pi = (2.0 / std::f32::consts::PI).sqrt();
    let coefficient = 0.044715;

    let inner = sqrt_2_over_pi
        * (x + coefficient * x.powi(3));

    let tanh_inner = inner.tanh();

    let sech_squared = 1.0 - tanh_inner.powi(2);

    0.5 * (1.0 + tanh_inner)
        + 0.5
            * x
            * sech_squared
            * sqrt_2_over_pi
            * (1.0 + 3.0 * coefficient * x.powi(2))
}


/// Numerically stable softmax.
///
/// Converts logits into probabilities whose sum is approximately 1.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }

    // Subtract maximum for numerical stability.
    let max_logit = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);

    let exp_values: Vec<f32> = logits
        .iter()
        .map(|&x| (x - max_logit).exp())
        .collect();

    let sum: f32 = exp_values.iter().sum();

    if sum == 0.0 {
        return vec![0.0; logits.len()];
    }

    exp_values
        .iter()
        .map(|&x| x / sum)
        .collect()
}