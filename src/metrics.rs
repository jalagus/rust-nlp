use std::ops::Mul;
use ndarray::Array2;
use ndarray_linalg::SVD;

pub fn frob_inner_product(mat1: &Array2<f32>, mat2: &Array2<f32>) -> f32 {
    mat1.mul(mat2).sum()
}

pub fn schatten(mat: &Array2<f32>, p: f32) -> f32 {
    let (_, s, _) = mat.svd(false, false).unwrap();
    s.map(|x| x.powf(p)).sum().powf(1.0 / p)
}

pub fn normalized_schatten(mat1: &Array2<f32>, mat2: &Array2<f32>, p: f32) -> f32 {
    let nom = schatten(&mat1.t().dot(mat2), p);
    let denom = (schatten(&mat1.t().dot(mat1), p) * schatten(&mat2.t().dot(mat2), p)).sqrt();

    nom / denom
}

pub fn normalized_frob(mat1: &Array2<f32>, mat2: &Array2<f32>) -> f32 {
    let nom = frob_inner_product(mat1, mat2);
    let denom = (frob_inner_product(mat1, mat1) * frob_inner_product(mat2, mat2)).sqrt();

    nom / denom
}


#[cfg(test)]
mod test {
    use ndarray::array;
    use super::*;
    
    #[test]
    fn normalized_frob_cos_with_self_is_one() {
        let test_mat = array![[1.0, 0.1],[1.5, -0.3]];
        assert_eq!(1.0, normalized_frob(&test_mat, &test_mat));
    }  

    #[test]
    fn normalized_schatten_cos_with_self_is_one() {
        let test_mat = array![[1.0, 0.1],[1.5, -0.3]];
        for p in 1..10 {
            assert_eq!(1.0, normalized_schatten(&test_mat, &test_mat, p as f32));
        }
    }  

}