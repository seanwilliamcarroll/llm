use llm::demo_train_codec;

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    assert_eq!(args.len(), 2, "Expected only one argument!");

    let filepath = args[1].clone();

    println!("Read from file: {filepath}");

    let text = std::fs::read_to_string(filepath)?;

    for additional_vocab in [0, 256, 768, 1280, 20278] {
        demo_train_codec(additional_vocab, &text);
    }
    Ok(())
}
