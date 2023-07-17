use std::collections::HashMap;
use flate2::read::GzDecoder;
use std::fs;
use std::io::prelude::*;
use ndarray::array;


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

fn encode(text: String, embeddings: HashMap<String, Vec<f32>>) {
    let tokens: Vec<&str> = text.split_whitespace().collect();

    let vectors: Vec<&Vec<f32>> = tokens.into_iter()
        .map(|x| embeddings.get(x).unwrap_or(vec![0.0])).collect();

    let arr = array![[1.0, 2.0], [2.0, 1.0]];
    println!("{:?}", vectors);
}

fn main() {
    //load_word2vec_gzip("google-word2vec.bin.gz".to_string());
    let embeddings = load_glove("glove.6B.50d.txt".to_string());
    encode(String::from("This is a test sentence."), embeddings);
    // Do the Kolmgorov thing with matrix SVDs for
    // https://aclanthology.org/2023.findings-acl.426.pdf
}
