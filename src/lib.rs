pub mod codec;

use codec::{BytePairEncodingCodec, BytePairEncodingCodecTrainer, Codec};
use std::time::Instant;

pub fn demo_codec(codec: &impl Codec, text: &str) {
    let start_time = Instant::now();

    let encoded = codec.encode(text);

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

#[must_use]
pub fn demo_train_codec(additional_vocab: usize, text: &str) -> BytePairEncodingCodec {
    println!();
    println!(
        "========================= Vocab size: {} =========================",
        (additional_vocab as u32) + codec::INITIAL_VOCABULARY_SIZE
    );

    println!("Original text: {} bytes", text.len());

    let mut codec_trainer = BytePairEncodingCodecTrainer::new();

    let start_time = Instant::now();

    codec_trainer.train(text, additional_vocab);

    let training_time = start_time.elapsed();

    println!("Training time: {training_time:.2?}");

    let codec = codec_trainer.get_codec();

    demo_codec(&codec, text);
    codec
}

pub fn demo_load_codec(filename: &str, text: &str) -> anyhow::Result<BytePairEncodingCodec> {
    let codec: BytePairEncodingCodec = BytePairEncodingCodec::load_from_file(filename)?;

    println!();
    println!(
        "========================= Vocab size: {} =========================",
        codec.vocab_size()
    );

    println!("Original text: {} bytes", text.len());

    demo_codec(&codec, text);

    Ok(codec)
}
