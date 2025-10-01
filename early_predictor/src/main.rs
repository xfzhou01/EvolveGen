use std::error::Error;
use ic3_param_predictor::Abc;
use std::env;
use std::process;
use std::path::{Path, PathBuf};
use std::fs::{create_dir_all, remove_file};
use sha2::Sha256;

mod feature_extraction;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    
    if args.len() > 1 && (args[1] == "-h" || args[1] == "--help") {
        println!("Usage: {} <input.aig> [output_dir]", args[0]);
        println!("  input.aig:   Input AIGER circuit file");
        println!("  output_dir:  Directory to store both feature files and graph data (default: benchmarks)");
        return Ok(());
    }
    
    if args.len() < 2 || args.len() > 3 {
        eprintln!("Usage: {} <input.aig> [output_dir]", args[0]);
        eprintln!("  output_dir:  Directory to store both feature files and graph data (default: benchmarks)");
        process::exit(1);
    }

    let input_path = Path::new(&args[1]);
    let output_dir = args.get(2).map(|s| Path::new(s)).unwrap_or_else(|| Path::new("benchmarks"));
    
    // Assert input file existence
    assert!(input_path.exists(), "Input AIGER file not found: {:?}", input_path);
    
    // Assert or create output directory
    assert!(output_dir.is_dir() || std::fs::create_dir_all(output_dir).is_ok(), "Failed to create output directory: {:?}", output_dir);
    
    // Create feature extractor and process circuit
    let mut extractor = feature_extraction::AigFeatureExtractor::new(
        input_path, 
        Some(output_dir), 
        Some(output_dir) // Use the same directory for both features and graph data
    );
    
    extractor.extract_all_features()?;
    
    Ok(())
}