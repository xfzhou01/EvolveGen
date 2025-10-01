use crate::Abc;
use std::path::{Path, PathBuf};
use std::fs::{self, File};
use serde_json;
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use anyhow::{Result, Context};
use serde::Serialize;
use std::io::Write;
use std::thread;
use std::time::Duration;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/*
For achieving the feature extraction, we need to use the ABC tool.

{
    "circuit_name": "6s8", ---> &r
    "static_features": {
      "size": {
        "num_pi": 86,  ----> &ps
        "num_flops": 396, -----> &ps
        "num_and_gates_aig": 3016 -----> &ps
      },
      "structure": {
        "num_mux_gates": 148, -----> &profile -m / &ps -m
        "num_xor_gates": 79, ----> &profile -a      
        "num_adders_total": 22 -----> &profile -a    
      },
      "depth": {
        "max_level": 65,      ----> &ps     
        "avg_level": 6.72     ----> &ps    
      },
      "connectivity": {
        "flop_fanout_variance": 14.09, -----> print_fanino -l
        "flop_fanout_stddev": 3.75 -------> print_fanino -l
      }
    }
  }

*/

/// Feature extractor for AIGER circuits
pub struct AigFeatureExtractor {
    circuit_path: PathBuf,
    work_dir: PathBuf,
    aig_graph_dir: Option<PathBuf>,  // Directory for PyG graph output
}

impl AigFeatureExtractor {
    /// Create new extractor with configurable work directory
    pub fn new<P: AsRef<Path>>(aig_path: P, feature_dir: Option<&Path>, aig_graph_dir: Option<&Path>) -> Self {
        // Assert input AIG file exists
        assert!(aig_path.as_ref().exists(), "AIG input file does not exist: {:?}", aig_path.as_ref());
        
        let case_name = Self::get_case_name(aig_path.as_ref());
        
        // Use provided feature directory or default to benchmarks/features
        let base_dir = feature_dir.unwrap_or_else(|| Path::new("benchmarks/features"));
        let work_dir = base_dir.join(case_name);
        
        // Convert aig_graph_dir to Option<PathBuf>
        let graph_dir = aig_graph_dir.map(|p| p.to_path_buf());
            
        Self {
            circuit_path: aig_path.as_ref().to_path_buf(),
            work_dir,
            aig_graph_dir: graph_dir,
        }
    }

    /// Extract all available features
    pub fn extract_all_features(&mut self) -> Result<AigFeatures> {
        // Create working directory
        fs::create_dir_all(&self.work_dir)
            .context("Failed to create feature directory")?;

        let mut abc = Abc::new();
        self.run_abc_commands(&mut abc)?;
        
        // Extract all features according to the example format
        let circuit_name = Self::get_case_name(&self.circuit_path);
        let static_features = self.extract_static_features(&mut abc)?;
        let dynamic_pdr = self.extract_dynamic_features(&mut abc)?;
        
        // Create features struct with the new format
        let features = AigFeatures::new(circuit_name.clone(), static_features, dynamic_pdr);

        // Save features to json in the work directory
        let json_path = self.work_dir.join("features.json");
        features.save_to_json(&json_path)?;
        
        // Also save the features JSON in Python-compatible format (circuit_name.json)
        if let Some(parent_dir) = self.work_dir.parent() {
            let py_json_path = parent_dir.join(format!("{}.json", circuit_name));
            features.save_to_json(&py_json_path)
                .context(format!("Failed to save Python-compatible JSON: {}", py_json_path.display()))?;
        }

        // If aig_graph_dir is set, generate PyG tensor data using PyO3
        // Files are already generated in run_abc_commands
        if let Some(ref graph_dir) = self.aig_graph_dir {
            let circuit_graph_dir = graph_dir.join(&circuit_name);
            
            // Try to generate PyG tensor object using PyO3
            let _ = self.generate_pyg_data(&circuit_graph_dir, &circuit_name);
        }

        Ok(features)
    }

    /// Execute ABC commands for feature extraction
    fn run_abc_commands(&self, abc: &mut Abc) -> Result<()> {
        abc.execute_command(&format!("read_aiger {}", self.circuit_path.display()));
        abc.execute_command("strash");
        
        // Detect FA/HA patterns
        let faha_path = self.work_dir.join("faha_detection.json");
        abc.execute_command("&get; &detect_faha -o detect_faha_output.json; &put");
        
        // Move the file to our target location
        if let Ok(content) = fs::read_to_string("detect_faha_output.json") {
            fs::write(&faha_path, content)?;
            // Assert FAHA detection output was created successfully
            assert!(faha_path.exists(), "FAHA detection output JSON not found after ABC command: {:?}", faha_path);
        }

        // Determine paths for edge list and node features
        // If aig_graph_dir is specified, generate PyG-compatible filenames
        let (el_path, feats_path) = if let Some(ref graph_dir) = self.aig_graph_dir {
            // Get circuit name for subfolder
            let circuit_name = Self::get_case_name(&self.circuit_path);
            let circuit_graph_dir = graph_dir.join(&circuit_name);
            
            // Create directory if it doesn't exist
            fs::create_dir_all(&circuit_graph_dir)
                .context("Failed to create graph directory")?;
                
            // Use PyG-compatible filenames
            (circuit_graph_dir.join("edge.csv"), circuit_graph_dir.join("node-feat.csv"))
        } else {
            // Use default filenames in work_dir
            (self.work_dir.join("circuit.el"), self.work_dir.join("node_feats.csv"))
        };
        
        let class_map_path = self.work_dir.join("class_map.json");
        
        let edgelist_cmd = format!(
            "&get; &edgelist -F {} -c {} -f {}", 
            el_path.display(), 
            class_map_path.display(), 
            feats_path.display()
        );
        
        abc.execute_command_with_output(&edgelist_cmd);
        
        // Assert that output files were created successfully
        assert!(el_path.exists(), "Edge list file not found after ABC command: {:?}", el_path);
        assert!(feats_path.exists(), "Node features file not found after ABC command: {:?}", feats_path);

        Ok(())
    }

    /// Run PDR (F=5, verbose) and collect dynamic stats JSON
    fn extract_dynamic_features(&self, abc: &mut Abc) -> Result<Option<JsonValue>> {
        // Ensure main network is in AIG world before running PDR
        abc.execute_command("&put");
        // Run PDR for 5 frames with verbose to trigger JSON dump in our patched ABC
        abc.execute_command("pdr -F 5 -v");

        // Look for the PDR JSON in several plausible locations
        let circuit_name = Self::get_case_name(&self.circuit_path);
        let candidates: [PathBuf; 4] = [
            PathBuf::from(format!("{}_pdr.json", circuit_name)),                                    // CWD: <name>_pdr.json
            PathBuf::from("pdr_stats.json"),                                                        // CWD: fallback name
            self.circuit_path.parent().unwrap_or(Path::new(".")).join(format!("{}_pdr.json", circuit_name)), // alongside input .aig
            self.circuit_path.parent().unwrap_or(Path::new(".")).join("pdr_stats.json"),                     // alongside input .aig (fallback)
        ];

        let src = candidates.iter().find(|p| p.exists()).cloned();
        let Some(src_path) = src else {
            return Ok(None);
        };

        // Move JSON into the work directory
        let dst = self.work_dir.join("pdr_stats.json");
        if src_path != dst {
            if let Err(e) = fs::rename(&src_path, &dst) {
                // Fallback: copy then remove original (e.g., cross-device move)
                let content = fs::read_to_string(&src_path)
                    .context(format!("Failed to read PDR JSON: {}", src_path.display()))?;
                fs::write(&dst, &content)
                    .context(format!("Failed to write PDR JSON to {}", dst.display()))?;
                let _ = fs::remove_file(&src_path);
            }
        }

        // Read from destination and parse
        let content = fs::read_to_string(&dst)
            .context(format!("Failed to read PDR JSON after move: {}", dst.display()))?;
        let json: JsonValue = serde_json::from_str(&content)
            .context("Failed to parse PDR JSON")?;

        Ok(Some(json))
    }

    /// Generate PyG compatible data using Python aig_to_pyg module
    fn generate_pyg_data(&self, circuit_graph_dir: &Path, circuit_name: &str) -> Result<()> {
        // Use PyO3 to call aig_to_pyg.py to generate PyG data directly
        // This is wrapped in a result to handle Python errors
        let py_result: PyResult<()> = Python::with_gil(|py| {
            // Try to import solver_tuner.data_utils.aig_to_pyg
            let aig_to_pyg = py.import("solver_tuner.data_utils.aig_to_pyg")?;
            
            // Call load_aig_as_pyg function
            let base_path = circuit_graph_dir.parent().unwrap_or(Path::new(""));
            let pyg_data = aig_to_pyg.call_method1(
                "load_aig_as_pyg", 
                (base_path.to_str().unwrap(), circuit_name)
            )?;
            
            // Cache the PyG data
            let cache_dir = circuit_graph_dir.join("cache");
            fs::create_dir_all(&cache_dir)?;
            
            aig_to_pyg.call_method1(
                "cache_aig_graph",
                (pyg_data, cache_dir.to_str().unwrap(), circuit_name)
            )?;
            
            Ok(())
        });
        
        // Convert PyO3 errors to anyhow errors
        if let Err(e) = py_result {
            println!("Warning: PyO3 error when generating PyG data: {}", e);
            println!("Graph files were generated successfully but PyG object could not be created.");
        }
            
        Ok(())
    }

    /// Get case name from file path
    fn get_case_name(path: &Path) -> String {
        path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string()
    }

    /// Extract static features according to example format
    fn extract_static_features(&self, abc: &mut Abc) -> Result<StaticFeatures> {
        let mut size = SizeFeatures::default();
        let mut depth = DepthFeatures::default();
        let mut structure = StructureFeatures::default();
        let mut connectivity = ConnectivityFeatures::default();

        // --- Parse from &ps ---
        // Example: ../examples/6s8 : i/o =     86/      1  ff =    396  and =    3016  lev =   65 (6.72)  mem = 0.05 MB
        let ps_output_str = abc.execute_command_with_output("&ps");
        //println!("DEBUG: &ps output:\n{}", ps_output_str);
        if let Some(line) = ps_output_str.lines().next() {
            //println!("DEBUG: First line of &ps: '{}'", line);
            // Parse PI
            if let Some(io_part) = line.split("i/o =").nth(1) { // "     86/      1  ff = ..."
                //println!("DEBUG: io_part: '{}'", io_part);
                let parts: Vec<&str> = io_part.split('/').collect();
                if parts.len() >= 1 {
                    size.num_pi = parts[0].trim().parse().unwrap_or(0); // "86"
                    //println!("DEBUG: num_pi parsed: {}", size.num_pi);
                }
            }
            // Parse Flops (ff)
            if let Some(ff_part) = line.split("ff =").nth(1) { // "    396  and = ..."
                //println!("DEBUG: ff_part: '{}'", ff_part);
                if let Some(ff_val_str) = ff_part.trim().split_whitespace().next() { // "396"
                    size.num_flops = ff_val_str.parse().unwrap_or(0);
                    //println!("DEBUG: num_flops parsed: {}", size.num_flops);
                }
            }
            // Parse AND gates (and) - fix parsing by using regex-like approach
            if let Some(and_part) = line.split("and =").nth(1) { // "    3016  lev = ..." or "    3016lev = ..."
                //println!("DEBUG: and_part: '{}'", and_part);
                // Extract digits at the beginning, handle case where there's no space before "lev"
                let and_val_str = and_part.trim()
                    .chars()
                    .take_while(|c| c.is_ascii_digit())
                    .collect::<String>();
                if !and_val_str.is_empty() {
                    size.num_and_gates_aig = and_val_str.parse().unwrap_or(0);
                    //println!("DEBUG: num_and_gates_aig parsed: {}", size.num_and_gates_aig);
                }
            }
            // Parse Depth (lev) - fix parsing similar to AND gates
            if let Some(lev_part) = line.split("lev =").nth(1) { // "   65 (6.72)  mem = ..." or "   65(6.72)..."
                //println!("DEBUG: lev_part: '{}'", lev_part);
                // Extract digits at the beginning for max_level
                let max_lev_str = lev_part.trim()
                    .chars()
                    .take_while(|c| c.is_ascii_digit())
                    .collect::<String>();
                if !max_lev_str.is_empty() {
                    depth.max_level = max_lev_str.parse().unwrap_or(0.0);
                    //println!("DEBUG: max_level parsed: {}", depth.max_level);
                }
                // Parse avg_level from parentheses
                if let Some(avg_part) = lev_part.split('(').nth(1) { // "6.72)  mem = ..." or "6.72)..."
                    if let Some(avg_lev_str) = avg_part.split(')').next() { // "6.72"
                        depth.avg_level = avg_lev_str.trim().parse().unwrap_or(0.0);
                        //println!("DEBUG: avg_level parsed: {}", depth.avg_level);
                    }
                }
            }
        }

        // --- Parse MUX gates and XOR gates from &ps -m ---
        // The output seems to be concatenated, need to find the XOR/MUX stats line
        let ps_m_output_str = abc.execute_command_with_output("&ps -m");
        //println!("DEBUG: &ps -m output:\n{}", ps_m_output_str);
        // Look for XOR/MUX stats anywhere in the output
        for line in ps_m_output_str.lines() {
            if line.contains("XOR/MUX stats:") || line.contains("mux =") {
                //println!("DEBUG: Found XOR/MUX stats line: '{}'", line);
                // Parse MUX gates
                if let Some(mux_stats_part) = line.split("mux =").nth(1) { // "     156  15.52 % ..."
                    //println!("DEBUG: mux_stats_part: '{}'", mux_stats_part);
                    if let Some(mux_val_str) = mux_stats_part.trim().split_whitespace().next() { // "156"
                        structure.num_mux_gates = mux_val_str.parse().unwrap_or(0);
                        //println!("DEBUG: num_mux_gates parsed: {}", structure.num_mux_gates);
                    }
                }
                // Parse XOR gates
                if let Some(xor_stats_part) = line.split("xor =").nth(1) { // "      79   7.86 %   mux = ..."
                    //println!("DEBUG: xor_stats_part: '{}'", xor_stats_part);
                    if let Some(xor_val_str) = xor_stats_part.trim().split_whitespace().next() { // "79"
                        structure.num_xor_gates = xor_val_str.parse().unwrap_or(0);
                        //println!("DEBUG: num_xor_gates parsed: {}", structure.num_xor_gates);
                    }
                }
            }
        }

        // --- Parse Connectivity from print_fanio -l  ---
        let fanio_output_str = abc.execute_command_with_output("print_fanio -l");
        //println!("DEBUG: print_fanio -l output:\n{}", fanio_output_str);
        
        if !fanio_output_str.trim().is_empty() {
            for line in fanio_output_str.lines() {
                if line.contains("Variance") || line.contains("variance") {
                    //println!("DEBUG: Found variance line in fanio: '{}'", line);
                    if let Some(val_str) = line.split('=').nth(1) {
                        connectivity.flop_fanout_variance = val_str.trim().parse().unwrap_or(0.0);
                        //println!("DEBUG: flop_fanout_variance parsed from fanio: {}", connectivity.flop_fanout_variance);
                    }
                }
                if line.contains("Standard deviation") || line.contains("standard deviation") {
                    //println!("DEBUG: Found stddev line in fanio: '{}'", line);
                    if let Some(val_str) = line.split('=').nth(1) {
                        connectivity.flop_fanout_stddev = val_str.trim().parse().unwrap_or(0.0);
                        //println!("DEBUG: flop_fanout_stddev parsed from fanio: {}", connectivity.flop_fanout_stddev);
                    }
                }
            }
        }

        // --- Parse num_adders_total from FA/HA detection ---
        if let Ok(num_adders) = self.parse_faha_features() {
            structure.num_adders_total = num_adders;
            //println!("DEBUG: num_adders_total parsed: {}", structure.num_adders_total);
        } else {
            //println!("DEBUG: Failed to parse FAHA features, using default value 0");
        }
        
        Ok(StaticFeatures {
            size,
            structure,
            depth,
            connectivity,
        })
    }

    /// Parse FA/HA detection results and return total number of adders
    fn parse_faha_features(&self) -> Result<u32> {
        let faha_path = self.work_dir.join("faha_detection.json");
        
        // Assert FAHA JSON exists for parsing
        assert!(faha_path.exists(), "FAHA JSON not found for parsing: {:?}", faha_path);
        
        let data = fs::read_to_string(faha_path)?;
        let json: serde_json::Value = serde_json::from_str(&data)?;

        // Assert JSON structure
        assert!(json["ha"].is_array(), "FAHA JSON 'ha' field must be an array");
        assert!(json["fa"].is_array(), "FAHA JSON 'fa' field must be an array");

        let chunk = |values: &Vec<i64>| {
            let mut chunks = Vec::new();
            let mut current = Vec::new();
            for &v in values {
                if v == 0 {
                    if !current.is_empty() {
                        chunks.push(current);
                        current = Vec::new();
                    }
                } else {
                    current.push(v);
                }
            }
            if !current.is_empty() {
                chunks.push(current);
            }
            chunks
        };

        // Parse HA/FA chunks
        let ha_values: Vec<i64> = json["ha"].as_array().unwrap().iter()
            .map(|v| v.as_i64().unwrap()).collect();
        let fa_values: Vec<i64> = json["fa"].as_array().unwrap().iter()
            .map(|v| v.as_i64().unwrap()).collect();

        let ha_chunks = chunk(&ha_values);
        let fa_chunks = chunk(&fa_values);

        // Total adders = Half adders + Full adders
        let num_adders_total = (ha_chunks.len() + fa_chunks.len()) as u32;

        Ok(num_adders_total)
    }
}

// Feature structs that match the example JSON format

#[derive(Debug, Default, Serialize)]
struct SizeFeatures {
    num_pi: u32,
    num_flops: u32,
    num_and_gates_aig: u32,
}

#[derive(Debug, Default, Serialize)]
struct StructureFeatures {
    num_mux_gates: u32,
    num_xor_gates: u32,
    num_adders_total: u32,
}

#[derive(Debug, Default, Serialize)]
struct DepthFeatures {
    max_level: f64,
    avg_level: f64,
}

#[derive(Debug, Default, Serialize)]
struct ConnectivityFeatures {
    flop_fanout_variance: f64,
    flop_fanout_stddev: f64,
}

#[derive(Debug, Serialize)]
struct StaticFeatures {
    size: SizeFeatures,
    structure: StructureFeatures,
    depth: DepthFeatures,
    connectivity: ConnectivityFeatures,
}

/// Container for all extracted features
#[derive(Debug, Serialize)]
pub struct AigFeatures {
    circuit_name: String,
    static_features: StaticFeatures,
    dynamic_features: Option<JsonValue>,
}

impl AigFeatures {
    /// Create new features container
    pub fn new(circuit_name: String, static_features: StaticFeatures, dynamic_features: Option<JsonValue>) -> Self {
        Self {
            circuit_name,
            static_features,
            dynamic_features,
        }
    }

    /// Save features to JSON format
    pub fn save_to_json<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path.as_ref())
            .context(format!("Failed to create file: {:?}", path.as_ref()))?;
        let mut writer = std::io::BufWriter::new(file);

        serde_json::to_writer_pretty(&mut writer, self)
            .context("Failed to serialize features to JSON")?;

        writer.flush().context("Failed to flush JSON writer")?;

        Ok(())
    }
} 