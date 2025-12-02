#[cfg(target_os = "macos")]
use anyhow::Result;

#[cfg(target_os = "macos")]
use dtln_rs::dtln_processor::{DtlnDeferredProcessor, DtlnProcessEngine};

#[cfg(target_os = "macos")]
use dtln_rs::dtln_utilities::{read_wav_to_pcm32, write_pcm32_to_wav};

#[cfg(target_os = "macos")]
const BLOCK_SIZE: usize = 1024;

#[cfg(target_os = "macos")]
const EXPECTED_SAMPLE_RATE: u32 = 16000;

// Build sample program that uses the dtln_rs library
// to process 16khz wav files for OSX.
#[cfg(target_os = "macos")]
fn main() -> Result<()> {
    // Check that there are two arguments
    if std::env::args().count() != 3 {
        println!("Usage: <input_wav_path> <output_wav_path>");
        std::process::exit(1);
    }
    // Get input name as first argument
    let input_name = std::env::args().nth(1).unwrap();

    check_is_wav(&input_name, true);

    // Get output name as second argument
    let output_name = std::env::args().nth(2).unwrap();
    check_is_wav(&output_name, false);

    let mut samples = vec![];
    let mut output = vec![];
    let _sample_rate = read_wav_to_pcm32(&input_name, &mut samples);
    let mut processor = DtlnDeferredProcessor::new()?;

    // Simulate blocked input for every 16834 samples
    for i in (0..samples.len()).step_by(BLOCK_SIZE) {
        if i + BLOCK_SIZE > samples.len() {
            std::thread::sleep(std::time::Duration::from_millis(10));
            output.append(&mut processor.denoise(&samples[i..])?.samples);
            break;
        }
        let mut denoise_result = processor.denoise(&samples[i..i + BLOCK_SIZE])?;

        if denoise_result.processor_starved {
            panic!("Processor starved");
        }

        // Sleep 30ms to simulate processing time
        std::thread::sleep(std::time::Duration::from_millis(10));

        output.append(&mut denoise_result.samples);
    }
    processor.stop();

    // Write to wav
    write_pcm32_to_wav(output, &output_name, EXPECTED_SAMPLE_RATE)?;
    Ok(())
}

#[cfg(target_os = "macos")]
fn check_is_wav(name: &str, check_exists: bool) {
    let path = std::path::Path::new(name);

    if check_exists {
        // Check that input_path exists and is a file
        let file = std::fs::File::open(path).unwrap();
        let file_metadata = file.metadata().unwrap();
        if !file_metadata.is_file() {
            panic!("File {} is not a file", name);
        }
    }

    // Check that it is a wav file
    let input_file_extension = path.extension().unwrap();
    if input_file_extension != "wav" {
        panic!("File {} is not a wav file", name);
    }
}

// Set up an empty main function for targets we don't provide a CLI for.
#[cfg(all(not(target_os = "macos"), not(target_os = "emscripten")))]
fn main() {}

// WebAssembly builds export functions without running a main.
#[cfg(target_os = "emscripten")]
fn main() {}

#[cfg(target_os = "emscripten")]
pub mod wasm;
