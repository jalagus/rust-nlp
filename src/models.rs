use std::collections::HashMap;
use ndarray::{Array1, Array2};
use ndarray_linalg::{TruncatedSvd, TruncatedOrder};
use std::ops::Mul;


pub fn encode(text: String, embeddings: &HashMap<String, Vec<f32>>) -> Array2<f32> {
    let tokens: Vec<&str> = text.split_whitespace().collect();
    let mut vectors: Vec<Vec<f32>> = Vec::new();

    for token in tokens {
        match embeddings.get(token) {
            Some(res) => vectors.push(res.to_vec()),
            None => {}
        }
    }

    let document_matrix: Array2<f32> = Array2::from_shape_vec(
        (vectors.len(), vectors[0].len()),
        vectors.iter().flatten().cloned().collect(),
    ).unwrap();

    document_matrix
}

pub fn lr_cov_repr(doc: &Array2<f32>, k: usize) -> Array2<f32> {
    let repr = TruncatedSvd::new(doc.t().dot(doc), TruncatedOrder::Largest)
        .decompose(k)
        .unwrap();

    let (u, s, _) = repr.values_vectors(); 
    let s_sqrt: Array1<f32> = s.iter().map(|x| x.sqrt()).collect();

    // return U * s^(1/2) 
    s_sqrt.broadcast((u.dim().0, k)).unwrap().mul(&u)
}