use clap::{ArgGroup, Parser};
use llm::demo_train_codec;

#[derive(Parser)]
#[command(group(
    ArgGroup::new("mode")
        .required(true)
        .args(["training_data", "load_file"])
))]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Input corpus file (*.txt)
    #[arg(short, long)]
    training_data: Option<String>,

    /// Optionally save trained codec to file
    #[arg(short, long, default_value = None, conflicts_with="load_file")]
    save_file: Option<String>,

    /// Load codec from existing file
    #[arg(short, long)]
    load_file: Option<String>,
}

fn main() -> std::io::Result<()> {
    let args = Cli::parse();

    if let Some(filepath) = args.training_data {
        println!("Read from file: {filepath}");

        let text = std::fs::read_to_string(filepath)?;

        for additional_vocab in [0, 256, 768, 1280, 20278] {
            demo_train_codec(additional_vocab, &text);
        }

        if let Some(_save_file) = args.save_file {
            todo!();
        }
    } else if let Some(_load_file) = args.load_file {
        todo!();
    }

    Ok(())
}
