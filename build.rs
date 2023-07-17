fn main() {
    // Tell Cargo that if the given file changes, to rerun this build script.
    println!("cargo:rustc-link-search=/usr/local/opt/openblas/lib");
    println!("cargo:rustc-link-arg=-lopenblas");
}