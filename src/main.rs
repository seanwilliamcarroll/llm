use std::collections::HashMap;
use std::convert::TryFrom;
use std::time::Instant;

#[derive(Debug, Hash, Eq, PartialEq, Clone, Copy, PartialOrd, Ord)]
struct Token(usize);

impl Token {
    fn as_byte(self) -> Option<u8> {
        u8::try_from(self.0).ok()
    }
}

impl std::fmt::Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if self.0 < 256usize {
            let character = char::from_u32(self.0 as u32).unwrap();
            if character.is_ascii() {
                return write!(f, "T<{}({:?})>", self.0, character);
            }
        }
        write!(f, "T<{}>", self.0)
    }
}

// impl Ord for Token {}

struct BytePairEncoder {
    encoding_rules: Vec<HashMap<Vec<u8>, Token>>,
    decoding_rules: HashMap<Token, Vec<u8>>,
    mapping: HashMap<(Token, Token), Token>,
    reverse_mapping: HashMap<Token, (Token, Token)>,
}

impl BytePairEncoder {
    fn new() -> Self {
        let rules = (0..=255u8).map(|value| (vec![value], Token(value as usize)));
        Self {
            encoding_rules: vec![rules.clone().collect::<HashMap<Vec<u8>, Token>>()],
            decoding_rules: rules
                .map(|(k, v)| (v, k))
                .collect::<HashMap<Token, Vec<u8>>>(),
            mapping: HashMap::new(),
            reverse_mapping: HashMap::new(),
        }
    }

    fn get_bytes(&self, token: Token) -> Vec<u8> {
        if let Some(output) = self.decoding_rules.get(&token) {
            output.clone()
        } else {
            // Need to build up from reverse mapping
            Self::tokens_to_bytes(self.expand_tokens(token))
                .expect("Do proper error checking here, but should just work")
        }
    }

    fn expand_tokens(&self, token: Token) -> Vec<Token> {
        let mut output = vec![];
        if let Some((token_a, token_b)) = self.reverse_mapping.get(&token) {
            output.extend(self.expand_tokens(*token_a).drain(0..));
            output.extend(self.expand_tokens(*token_b).drain(0..));
        } else {
            output.push(token);
        }
        output
    }

    fn tokens_to_bytes(tokens: Vec<Token>) -> Option<Vec<u8>> {
        tokens
            .into_iter()
            .map(Token::as_byte)
            .collect::<Option<Vec<u8>>>()
    }

    fn add_encoding_rule(&mut self, pair: (Token, Token), new_token: Token) {
        assert_eq!(self.mapping.insert(pair, new_token), None);
        assert_eq!(self.reverse_mapping.insert(new_token, pair), None);

        let bytes_key = self.get_bytes(new_token);

        assert!(bytes_key.len() > 1);

        if self.encoding_rules.len() < bytes_key.len() {
            self.encoding_rules.extend(vec![
                HashMap::new();
                bytes_key.len() - self.encoding_rules.len()
            ]);
        }
        self.encoding_rules[bytes_key.len() - 1].insert(bytes_key.clone(), new_token);
        self.decoding_rules.insert(new_token, bytes_key);
    }

    fn find_next_token(&self, slice: &[u8]) -> (Token, usize) {
        let max_encoding_size = usize::min(self.encoding_rules.len(), slice.len());
        for encoding_table_index in (0..max_encoding_size).rev() {
            let encoding_size = encoding_table_index + 1;
            let encoding_slice = &slice[0..encoding_size];
            if let Some(token) = self.encoding_rules[encoding_table_index].get(encoding_slice) {
                return (*token, encoding_size);
            }
        }
        panic!("Didn't find next token!")
    }

    fn encode(&self, input: &str) -> Vec<Token> {
        let mut output = vec![];

        let raw_bytes = input.bytes().collect::<Vec<u8>>();

        let mut index: usize = 0;

        let max_encoding_size = self.encoding_rules.len();

        while index < raw_bytes.len() {
            let end_bound = if index + max_encoding_size > raw_bytes.len() {
                raw_bytes.len()
            } else {
                index + max_encoding_size
            };
            let encoding_slice = &raw_bytes[index..end_bound];
            let (next_token, encoding_size) = self.find_next_token(encoding_slice);
            output.push(next_token);
            index += encoding_size;
        }

        output
    }

    fn decode(&self, tokens: &[Token]) -> String {
        String::from_utf8(
            tokens
                .iter()
                .flat_map(|token| self.get_bytes(*token))
                .collect::<Vec<u8>>(),
        )
        .expect("Shouldn't fail")
    }

    fn frequency_count(&self, tokens: &[Token]) -> Vec<(usize, Token)> {
        let mut counts: HashMap<Token, usize> = HashMap::new();

        for token in tokens {
            *counts.entry(*token).or_insert(0) += 1;
        }

        let mut output = counts.into_iter().map(|(k, v)| (v, k)).collect::<Vec<_>>();

        output.sort();
        output.reverse();
        output
    }

    fn print_vocab(&self, tokens: &[Token]) {
        let frequency_counts = self.frequency_count(tokens);

        for (count, token) in frequency_counts.into_iter().take(5) {
            println!("Top 5 Tokens");
            print!("{token} ({count} times)");
            let bytes = self.decoding_rules.get(&token).unwrap();
            if let Ok(string_token) = String::from_utf8(bytes.clone()) {
                println!(" \"{string_token}\"");
            } else {
                println!();
            }
        }
        println!();
    }
}

#[allow(dead_code)]
struct Tokenizer {
    current_token_id: usize,
    encoder: BytePairEncoder,
}

#[allow(dead_code)]
impl Tokenizer {
    fn new() -> Self {
        Self {
            current_token_id: 256usize,
            encoder: BytePairEncoder::new(),
        }
    }

    fn encode(&self, input: &str) -> Vec<Token> {
        self.encoder.encode(input)
    }

    fn decode(&self, tokens: &[Token]) -> String {
        self.encoder.decode(tokens)
    }

    fn count_pairs(input: &[Token]) -> HashMap<(Token, Token), usize> {
        let mut output = HashMap::new();

        for pair in input.windows(2) {
            let pair = match pair {
                [a, b] => (*a, *b),
                _ => panic!("Can't happen"),
            };
            *output.entry(pair).or_insert(0) += 1;
        }

        output
    }

    fn print_vocab(&self, tokens: &[Token]) {
        self.encoder.print_vocab(tokens);
    }

    fn train(&mut self, input: &str, additional_merges: usize) {
        let mut tokens = input
            .bytes()
            .map(|byte| Token(byte as usize))
            .collect::<Vec<Token>>();

        for added_vocab in 0..additional_merges {
            if added_vocab == 2000 {
                tokens = tokens.into_iter().take(200_000).collect::<Vec<Token>>();
            }

            let pairs = Self::count_pairs(&tokens);

            let top_pairs = {
                let mut top_pairs = pairs.into_iter().collect::<Vec<_>>();
                top_pairs.sort_by_key(|(_, val)| std::cmp::Reverse(*val));
                top_pairs.into_iter().take(10).collect::<Vec<_>>()
            };

            let (top_pair, _) = top_pairs[0];

            let new_tokens = {
                let mut new_tokens = Vec::with_capacity(tokens.len());

                tokens.reverse();

                let mut current_tokens = Vec::with_capacity(2);
                while let Some(next_token) = tokens.pop() {
                    current_tokens.push(next_token);
                    match &current_tokens[..] {
                        [token_a, token_b] => {
                            let token_a = *token_a;
                            let token_b = *token_b;

                            if (token_a, token_b) == top_pair {
                                new_tokens.push(Token(self.current_token_id));
                                current_tokens.clear();
                            } else {
                                new_tokens.push(token_a);
                                current_tokens.remove(0);
                            }
                        }
                        [_] => {}
                        _ => panic!("Can't happen"),
                    }
                }
                new_tokens.extend(current_tokens);
                new_tokens
            };
            self.encoder
                .add_encoding_rule(top_pair, Token(self.current_token_id));
            self.current_token_id += 1;
            tokens = new_tokens;
        }
    }
}

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

        let mut tokenizer = Tokenizer::new();

        let start_time = Instant::now();

        tokenizer.train(&text, additional_vocab);

        let training_time = start_time.elapsed();

        println!("Training time: {training_time:.2?}");

        let start_time = Instant::now();

        let encoded = tokenizer.encode(&text);

        let encoding_time = start_time.elapsed();

        println!(
            "Encoded tokens {} ({:.2}x)",
            encoded.len(),
            (text.len() as f64) / (encoded.len() as f64),
        );
        println!("Encoding time: {encoding_time:.2?}");

        let start_time = Instant::now();

        let decoded = tokenizer.decode(&encoded);

        let decoding_time = start_time.elapsed();

        println!("Decoded text {} bytes", decoded.len());
        assert_eq!(text.len(), decoded.len());

        println!("Decoding time: {decoding_time:.2?}");

        tokenizer.print_vocab(&encoded);
    }
    Ok(())
}
