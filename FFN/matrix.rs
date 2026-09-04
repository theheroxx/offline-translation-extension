use burn::prelude::*;


pub fn matrix_mul<B: Backend>(
    a: Tensor<B, 2>,
    b: Tensor<B, 2>,
) -> Tensor<B, 2> {
    a.matmul(b)
}