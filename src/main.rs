use std::{collections::HashMap, ops::Mul, str::FromStr, string::ParseError};
use flate2::read::GzDecoder;
use std::fs;
use std::io::prelude::*;
use ndarray::{Array1, Array2};
use ndarray_linalg::{TruncatedSvd, TruncatedOrder};
use structopt::StructOpt;
use std::path::Path;


enum EmbeddingType {
    Word2Vec,
    Glove
}

impl FromStr for EmbeddingType {
    type Err = ParseError;
    fn from_str(day: &str) -> Result<Self, Self::Err> {
        match day {
            "w2v" => Ok(EmbeddingType::Word2Vec),
            "glove" => Ok(EmbeddingType::Glove),
            _ => Ok(EmbeddingType::Glove) // TODO: Maybe should be an error
        }
    }    
}

#[derive(StructOpt)]
struct Options {
    #[structopt(default_value="glove.dev.50d.txt")]
    /// Embeddings file
    emb_file: String,
    #[structopt()]
    /// Type of embeddings
    emb_type: EmbeddingType
}


fn load_word2vec_gzip(file_path: &Path) -> HashMap<String, Vec<f32>> {
    let gzip_file = fs::File::open(file_path.as_os_str()).unwrap();
    let mut decompressed = GzDecoder::new(gzip_file);
    let mut buffer = [0; 18];
    decompressed.read(&mut buffer).unwrap();
    //decompressed.read_to_string(&mut s).unwrap();

    //let words: Vec<String> = s.split("\n").map(|x| x.to_string()).collect();
    println!("{:?}", std::str::from_utf8(&buffer).unwrap());

    HashMap::new()
}

fn load_glove(file_path: &Path) -> HashMap<String, Vec<f32>> {
    let mut file = fs::File::open(file_path).unwrap();
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

fn encode(text: String, embeddings: &HashMap<String, Vec<f32>>) -> Array2<f32> {
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

fn lr_cov_repr(doc: &Array2<f32>, k: usize) -> Array2<f32> {
    let repr = TruncatedSvd::new(doc.t().dot(doc), TruncatedOrder::Largest)
        .decompose(k)
        .unwrap();

    let (u, s, _) = repr.values_vectors(); 
    let s_sqrt: Array1<f32> = s.iter().map(|x| x.sqrt()).collect();

    // return U * s^(1/2) 
    s_sqrt.broadcast((u.dim().0, k)).unwrap().mul(&u)
}

fn frob_inner_product(mat1: &Array2<f32>, mat2: &Array2<f32>) -> f32 {
    mat1.mul(mat2).sum()
}

fn normalized_frob(mat1: &Array2<f32>, mat2: &Array2<f32>) -> f32 {
    let nom = frob_inner_product(mat1, mat2);
    let denom = (frob_inner_product(mat1, mat1) * frob_inner_product(mat2, mat2)).sqrt();

    nom/denom
}

fn main() {
    let options = Options::from_args();
    let emb_path = Path::new(&options.emb_file);

    let embeddings = match options.emb_type {
        EmbeddingType::Glove => load_glove(emb_path),
        EmbeddingType::Word2Vec => load_word2vec_gzip(emb_path),
    };

    let doc = encode(String::from("with the said with by his from said"), &embeddings);
    let doc2 = encode(String::from("the the the said with by his from the said"), &embeddings);
    let repr = lr_cov_repr(&doc, 2);
    let repr2 = lr_cov_repr(&doc2, 2);

    println!("{:?}, {:?}", repr.dim(), repr2.dim());
    println!("{:?}", normalized_frob(&repr, &repr2));

    // Do the Kolmgorov thing with matrix SVDs for
    // https://aclanthology.org/2023.findings-acl.426.pdf
}
