pub fn matrix_mul(a: &[Vec<f32>], b: &[Vec<f32>]) -> Vec<Vec<f32>> {
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
