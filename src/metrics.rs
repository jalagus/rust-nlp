use std::ops::Mul;
use ndarray::Array2;

pub fn frob_inner_product(mat1: &Array2<f32>, mat2: &Array2<f32>) -> f32 {
    mat1.mul(mat2).sum()
}

pub fn normalized_frob(mat1: &Array2<f32>, mat2: &Array2<f32>) -> f32 {
    let nom = frob_inner_product(mat1, mat2);
    let denom = (frob_inner_product(mat1, mat1) * frob_inner_product(mat2, mat2)).sqrt();

    nom/denom
}