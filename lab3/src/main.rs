// use peroxide::prelude::*;
use peroxide::fuga::*;

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

fn matrix_cond_1(a: &Matrix) -> f64 {
    matrix_norm_1(a) * matrix_norm_1(&a.inv())
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

fn matrix_cond_inf(a: &Matrix) -> f64 {
    matrix_norm_inf(a) * matrix_norm_inf(&a.inv())
}

fn matrix_norm_2(a: &Matrix) -> f64 {
    let eigenvalues = eigen(&(&a.t() * a), EigenMethod::Jacobi).eigenvalue;
    let mut max = f64::NEG_INFINITY;
    for x in eigenvalues {
        if x > max {
            max = x;
        }
    }
    max.sqrt()
}

fn matrix_cond_2(a: &Matrix) -> f64 {
    matrix_norm_2(a) * matrix_norm_2(&a.inv())
}

fn matrix_norm_p(a: &Matrix, p: f64) -> f64 {
    let eig = eigen(&a, EigenMethod::Jacobi);
    let mut sum = 0.0;
    for x in eig.eigenvalue {
        sum += x.pow(p);
    }
    sum.pow(1.0 / p)
}

fn matrix_cond_p(a: &Matrix, p: f64) -> f64 {
    matrix_norm_p(a, p) * matrix_norm_p(&a.inv(), p)
}

fn main() {
    let a = ml_matrix("4 9 2; 3 5 7; 8 1 6");

    let expected = Normed::norm(&a, Norm::L1);
    let result = matrix_norm_1(&a);
    assert_close!(expected, result);
    println!("norm_1: {}", result);
    println!("cond_1: {}", matrix_cond_1(&a));

    let expected = Normed::norm(&a, Norm::L2).sqrt();
    let result = matrix_norm_2(&a);
    assert_close!(expected, result);
    println!("norm_2: {}", result);
    println!("cond_2: {}", matrix_cond_2(&a));

    let expected = Normed::norm(&a, Norm::LInf);
    let result = matrix_norm_inf(&a);
    assert_close!(expected, result);
    println!("norm_inf: {}", result);
    println!("cond_inf: {}", matrix_cond_inf(&a));

    let result = matrix_norm_p(&a, 15.0);
    println!("norm_p15: {}", result);
    println!("cond_p15: {}", matrix_cond_p(&a, 15.0));

    let result = a.svd();
    println!("SVD: {:?}", result);

}
