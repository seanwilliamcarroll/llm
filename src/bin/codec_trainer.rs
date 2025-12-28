use clap::{ArgGroup, Parser};
use llm::{codec::Codec, demo_load_codec, demo_train_codec};
use std::path::Path;

#[derive(Parser)]
#[command(group(
    ArgGroup::new("mode")
        .args(["save_file_base", "load_file"])
))]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Input corpus file (*.txt)
    training_data: String,

    #[allow(clippy::doc_markdown)]
    /// Optionally save trained codecs to file, in format
    ///
    ///     <SAVE_FILE_BASE>_<TRAINING_DATA>_<VOCAB_SIZE>.cdx
    ///
    /// with <TRAINING_DATA> equal to the base filename of training-data
    #[arg(short, long, default_value = None, verbatim_doc_comment)]
    save_file_base: Option<String>,

    /// Load codec from existing file
    #[arg(short, long)]
    load_file: Option<String>,
}

fn main() -> std::io::Result<()> {
    let args = Cli::parse();

    let filepath = args.training_data;
    println!("Read from file: {filepath}");

    let text = std::fs::read_to_string(filepath.clone())?;

    if let Some(load_file) = args.load_file {
        match demo_load_codec(&load_file, &text) {
            Ok(_) => (),
            Err(error) => {
                eprintln!("Failed to load file: {load_file}\n{error}");
                std::process::exit(1);
            }
        };
    } else {
        for additional_vocab in [0, 256, 768, 1280, 20278] {
            let codec = demo_train_codec(additional_vocab, &text);
            if let Some(save_file) = args.save_file_base.as_ref() {
                // Save the last codec
                let new_filename = format!(
                    "{save_file}_{}_{}.cdx",
                    Path::new(&filepath)
                        .file_stem()
                        .expect("Already read this file")
                        .to_str()
                        .unwrap_or("training_data"),
                    codec.vocab_size()
                );
                match codec.save_to_file(&new_filename) {
                    Ok(()) => {
                        println!("Saved to file: {new_filename}");
                    }
                    Err(error) => {
                        eprintln!("Failed to save to file: {new_filename}\n{error}");
                        std::process::exit(1);
                    }
                }
            }
        }
    }

    Ok(())
}
