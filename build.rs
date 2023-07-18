fn main() {
    println!("cargo:rustc-link-search=/usr/local/opt/openblas/lib");
    println!("cargo:rustc-link-arg=-lopenblas");
}