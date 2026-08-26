// mod gradient {
    use crate::input_module::Labels;
    use crate::softmax;
//     pub struct Gradients {
//     d_weights: Vec<Vec<f32>>,
//     d_biases: Vec<f32>,
// }


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
// }

