pub fn relu(x: f32) -> f32 {
    x.max(0.0)
}


pub fn apply_relu(values: &[f32]) -> Vec<f32> {
    values.iter().map(|&x| relu(x)).collect()
}


pub fn gelu(x: f32) -> f32 {
    let sqrt_2_over_pi = (2.0 / std::f32::consts::PI).sqrt();

    0.5 * x
        * (1.0
            + (sqrt_2_over_pi * (x + 0.044715 * x.powi(3))).tanh())
}

pub fn apply_gelu(values: &[f32]) -> Vec<f32> {
    values.iter()
        .map(|&x| gelu(x))
        .collect()
}


pub fn softmax(logits: &[f32]) -> Vec<f32> {
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

