use peroxide::prelude::*;
use peroxide::fuga;

macro_rules! assert_close {
    ($expected:expr, $result:expr) => {
        let eps = 1e-10;
        assert!(($expected - $result).abs() < eps, "Expected: {}, Result: {}", $expected, $result);
    };
}

fn matrix_norm_1(a: &Matrix) -> f64 {
    let mut max_sum = f64::NEG_INFINITY;
    for j in 0..a.col {
        let mut sum = 0f64;
        for i in 0..a.row {
            sum += a[(i, j)].abs();
        }
        max_sum = max_sum.max(sum);
    }
    max_sum
}

fn matrix_norm_inf(a: &Matrix) -> f64 {
    let mut max_sum = f64::NEG_INFINITY;
    for i in 0..a.row {
        let mut sum = 0f64;
        for j in 0..a.col {
            sum += a[(i, j)].abs();
        }
        max_sum = max_sum.max(sum);
    }
    max_sum
}

fn matrix_norm_2(a: &Matrix) -> f64 {
    let eigenvalues = eigen(&(&a.t() * a)).eigenvalue;
    let mut max = f64::NEG_INFINITY;
    for x in eigenvalues {
        if x > max {
            max = x;
        }
    }
    max // .sqrt() ???
}

fn main() {
    let a = rand(8, 8);

    let expected = fuga::Normed::norm(&a, fuga::Norm::L1);
    let result = matrix_norm_1(&a);
    assert_close!(expected, result);

    let expected = fuga::Normed::norm(&a, fuga::Norm::LInf);
    let result = matrix_norm_inf(&a);
    assert_close!(expected, result);

    let expected = fuga::Normed::norm(&a, fuga::Norm::L2);
    let result = matrix_norm_2(&a);
    assert_close!(expected, result);
}
