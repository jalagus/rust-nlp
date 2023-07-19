use std::{str::FromStr, string::ParseError};
use structopt::StructOpt;
use std::path::Path;

mod metrics;
mod models;
mod utils;

enum EmbeddingType {
    Word2Vec,
    Glove
}

impl FromStr for EmbeddingType {
    type Err = ParseError;
    fn from_str(embedding_type: &str) -> Result<Self, Self::Err> {
        match embedding_type {
            "w2v" => Ok(EmbeddingType::Word2Vec),
            "glove" => Ok(EmbeddingType::Glove),
            _ => todo!("Should prbably return an error.")
        }
    }    
}

#[derive(StructOpt)]
struct Options {
    #[structopt(default_value="glove.dev.50d.txt")]
    /// Embeddings file
    emb_file: String,
    #[structopt(default_value="glove")]
    /// Type of embeddings
    emb_type: EmbeddingType
}

fn main() {
    let options = Options::from_args();
    let emb_path = Path::new(&options.emb_file);

    let embeddings = match options.emb_type {
        EmbeddingType::Glove => utils::load_glove(emb_path),
        EmbeddingType::Word2Vec => utils::load_word2vec_gzip(emb_path),
    };

    let doc = models::encode(String::from("with the said with by his from said"), &embeddings);
    let doc2 = models::encode(String::from("the the the said with by his from the said"), &embeddings);
    let repr = models::lr_cov_repr(&doc, 2);
    let repr2 = models::lr_cov_repr(&doc2, 2);

    println!("{:?}, {:?}", repr.dim(), repr2.dim());
    println!("{:?}", metrics::normalized_frob(&repr, &repr2));

    // Do the Kolmgorov thing with matrix SVDs for
    // https://aclanthology.org/2023.findings-acl.426.pdf
}
