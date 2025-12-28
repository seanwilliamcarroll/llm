mod codec;

use codec::{BytePairEncodingCodecTrainer, Codec};
use std::time::Instant;

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    assert_eq!(args.len(), 2, "Expected only one argument!");

    let filepath = args[1].clone();

    println!("Read from file: {filepath}");

    let text = std::fs::read_to_string(filepath)?;

    for additional_vocab in [0, 256, 768, 1280, 20278] {
        println!();
        println!(
            "========================= Vocab size: {} =========================",
            additional_vocab + 256
        );

        println!("Original text: {} bytes", text.len());

        let mut codec_trainer = BytePairEncodingCodecTrainer::new();

        let start_time = Instant::now();

        codec_trainer.train(&text, additional_vocab);

        let training_time = start_time.elapsed();

        println!("Training time: {training_time:.2?}");

        let codec = codec_trainer.get_codec();

        let start_time = Instant::now();

        let encoded = codec.encode(&text);

        let encoding_time = start_time.elapsed();

        println!(
            "Encoded tokens {} ({:.2}x)",
            encoded.len(),
            (text.len() as f64) / (encoded.len() as f64),
        );
        println!("Encoding time: {encoding_time:.2?}");

        let start_time = Instant::now();

        let decoded = codec.decode(&encoded);

        let decoding_time = start_time.elapsed();

        println!("Decoded text {} bytes", decoded.len());
        assert_eq!(text.len(), decoded.len());

        println!("Decoding time: {decoding_time:.2?}");

        codec.print_vocab(&encoded);
    }
    Ok(())
}
