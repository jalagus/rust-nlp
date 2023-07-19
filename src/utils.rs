use flate2::read::GzDecoder;
use std::fs;
use std::io::prelude::*;
use std::collections::HashMap;
use std::path::Path;


pub fn load_word2vec_gzip(file_path: &Path) -> HashMap<String, Vec<f32>> {
    let gzip_file = fs::File::open(file_path.as_os_str()).unwrap();
    let mut decompressed = GzDecoder::new(gzip_file);
    let mut buffer = [0; 18];
    decompressed.read(&mut buffer).unwrap();
    //decompressed.read_to_string(&mut s).unwrap();

    //let words: Vec<String> = s.split("\n").map(|x| x.to_string()).collect();
    println!("{:?}", std::str::from_utf8(&buffer).unwrap());

    HashMap::new()
}

pub fn load_glove(file_path: &Path) -> HashMap<String, Vec<f32>> {
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