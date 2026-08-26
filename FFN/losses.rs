use crate::FFN::input_module::Labels;

pub fn mse_loss(y: &Labels, y_pred: &[f32]) -> f32 {
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

pub fn cross_entropy_loss(y: &Labels, y_pred: &[f32]) -> f32 {
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
