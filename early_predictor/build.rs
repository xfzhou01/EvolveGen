use std::env;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let src_dir = env::var("CARGO_MANIFEST_DIR")
        .map_err(|e| e.to_string())?;

    println!(
        "cargo:rustc-link-search=native={}",
        PathBuf::from(src_dir).display()
    );

    // Link against required system libraries first
    println!("cargo:rustc-link-lib=dylib=readline");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=m");
    println!("cargo:rustc-link-lib=dylib=dl");
    println!("cargo:rustc-link-lib=dylib=pthread");
    println!("cargo:rustc-link-lib=dylib=z");

    // Link against the static ABC library with whole-archive
    println!("cargo:rerun-if-changed=./libabc.a");
    println!("cargo:rustc-link-arg=-L.");
    println!("cargo:rustc-link-arg=-Wl,--whole-archive");
    println!("cargo:rustc-link-arg=-l:libabc.a");
    println!("cargo:rustc-link-arg=-Wl,--no-whole-archive");

    // Add linker arguments
    println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
    println!("cargo:rustc-link-arg=-Wl,--start-group");
    println!("cargo:rustc-link-arg=-lstdc++");
    println!("cargo:rustc-link-arg=-lreadline");
    println!("cargo:rustc-link-arg=-lm");
    println!("cargo:rustc-link-arg=-ldl");
    println!("cargo:rustc-link-arg=-Wl,--end-group");

    // Add gRPC build steps
    // tonic_build::compile_protos("../grpc_communicator/proto/service.proto")?;

    Ok(())
}