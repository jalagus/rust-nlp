use std::collections::HashMap;
use flate2::read::GzDecoder;
use std::fs;
use std::io::prelude::*;
use ndarray::{Array1, Array2};


fn load_word2vec_gzip(filename: String) -> HashMap<String, Vec<f32>> {
    let gzip_file = fs::File::open(filename).unwrap();
    let mut decompressed = GzDecoder::new(gzip_file);
    let mut buffer = [0; 18];
    decompressed.read(&mut buffer).unwrap();
    //decompressed.read_to_string(&mut s).unwrap();

    //let words: Vec<String> = s.split("\n").map(|x| x.to_string()).collect();
    println!("{:?}", std::str::from_utf8(&buffer).unwrap());

    HashMap::new()
}

fn load_glove(filename: String) -> HashMap<String, Vec<f32>> {
    let mut file = fs::File::open(filename).unwrap();
    let mut s = String::new();

    file.read_to_string(&mut s).expect("Could not read file.");

    let words: HashMap<String, Vec<f32>> = s.split("\n").map(|x| {
        let row: Vec<&str> = x.split_whitespace().collect();
        if row.len() > 0 {
            let vector: Vec<f32> = (&row[1..]).iter().map(|v| v.parse::<f32>().unwrap()).collect();
            return (row[0].to_string(), vector);
        }
        ("<UNK>".to_string(), vec![0.0])
    }).collect();

    words
}

fn encode(text: String, embeddings: HashMap<String, Vec<f32>>) -> Array2<f32> {
    let tokens: Vec<&str> = text.split_whitespace().collect();
    let mut vectors: Vec<Vec<f32>> = Vec::new();

    for token in tokens {
        match embeddings.get(token) {
            Some(res) => vectors.push(res.to_vec()),
            None => {}
        }
    }

    let ndarray_2d: Array2<f32> = Array2::from_shape_vec(
        (vectors.len(), vectors[0].len()),
        vectors.iter().flatten().cloned().collect(),
    ).unwrap();

    ndarray_2d
}

fn lr_cov_repr(doc: &Array2<f32>, k: usize) -> (Array2<f32>, Array1<f32>, Array2<f32>) {
    let repr = ndarray_linalg::TruncatedSvd::new(doc.t().dot(doc), ndarray_linalg::TruncatedOrder::Largest)
        .decompose(k)
        .unwrap();

    let (u, s, v) = repr.values_vectors();
    
    let s_sqrt: Array1<f32> = s.iter().map(|x| x.sqrt()).collect();

    (u, s, v)
}

fn main() {
    //load_word2vec_gzip("google-word2vec.bin.gz".to_string());
    let embeddings = load_glove("glove.dev.50d.txt".to_string());
    let doc = encode(String::from("the said with by his from"), embeddings);

    let (u, s, v) = lr_cov_repr(&doc, 2);

    println!("Singular values: {:?}", s);
    // Do the Kolmgorov thing with matrix SVDs for
    // https://aclanthology.org/2023.findings-acl.426.pdf
}
