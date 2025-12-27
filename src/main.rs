use std::collections::HashMap;
use std::convert::TryFrom;

#[derive(Debug, Hash, Eq, PartialEq, Clone, Copy)]
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

#[allow(dead_code)]
struct Tokenizer {
    current_token_id: usize,
    tokens: Vec<Token>,
    // encoder: Encoder,
    // decoder: Decoder,
    mapping: HashMap<(Token, Token), Token>,
    reverse_mapping: HashMap<Token, (Token, Token)>,
}

struct BytePairEncoder {
    encoding_rules: Vec<HashMap<Vec<u8>, Token>>,
    decoding_rules: HashMap<Token, Vec<u8>>,
    mapping: HashMap<(Token, Token), Token>,
    reverse_mapping: HashMap<Token, (Token, Token)>,
}

impl BytePairEncoder {
    fn new() -> Self {
        let rules = (0..256).map(|value| (vec![value as u8], Token(value)));
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
            .map(|token| token.as_byte())
            .collect::<Option<Vec<u8>>>()
    }

    fn add_encoding_rule(&mut self, pair: (Token, Token), new_token: Token) {
        assert_eq!(self.mapping.insert(pair, new_token), None);
        assert_eq!(self.reverse_mapping.insert(new_token, pair), None);

        let bytes_key = self.get_bytes(new_token);

        // If we're adding a new rule,
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

    fn encode(&self, input: String) -> Vec<Token> {
        let mut output = vec![];

        let raw_bytes = input.into_bytes();

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
            index = index + encoding_size;
        }

        output
    }

    fn decode(&self, tokens: Vec<Token>) -> String {
        String::from_utf8(
            tokens
                .into_iter()
                .map(|token| self.get_bytes(token))
                .flatten()
                .collect::<Vec<u8>>(),
        )
        .expect("Shouldn't fail")
    }
}

#[allow(dead_code)]
impl Tokenizer {
    fn new() -> Self {
        Self {
            current_token_id: 256usize,
            tokens: (0..256).map(|value| Token(value)).collect::<Vec<_>>(),
            mapping: HashMap::new(),
            reverse_mapping: HashMap::new(),
            // encoder: Encoder::new(),
            // decoder: Decoder::new(),
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

    fn tokens_to_string(&self, tokens: Vec<Token>) -> String {
        let mut output: Vec<u8> = vec![];
        for token in tokens {
            // Recursively expand tokens

            for token in self.expand_tokens(token) {
                assert!(token.0 < 256usize);
                let byte: u8 = token.0 as u8;
                output.push(byte);
            }
        }
        String::from_utf8(output).expect("Must work")
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

    fn train(&mut self, input: String) {
        let mut tokens = input
            .bytes()
            .map(|byte| Token(byte as usize))
            .collect::<Vec<Token>>();

        println!(
            "First tokens: {:?}",
            tokens.iter().take(10).collect::<Vec<_>>()
        );

        let original_len = tokens.len() as f64;

        println!("Before: {}", tokens.len());

        for _ in 0..256 {
            let pairs = Self::count_pairs(&tokens);

            let top_pairs = {
                let mut top_pairs = pairs.into_iter().collect::<Vec<_>>();
                top_pairs.sort_by_key(|(_, val)| std::cmp::Reverse(*val));
                top_pairs.into_iter().take(10).collect::<Vec<_>>()
            };

            let (top_pair, _) = top_pairs[0];

            // println!("\n\nTop pairs:");
            // for ((token_a, token_b), num) in top_pairs {
            //     println!("{token_a}, {token_b} = {num} times");
            // }

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
            self.mapping.insert(top_pair, Token(self.current_token_id));
            self.reverse_mapping
                .insert(Token(self.current_token_id), top_pair);
            self.current_token_id += 1;
            tokens = new_tokens;
            println!(
                "After: {} ({:.2}x)",
                tokens.len(),
                original_len / (tokens.len() as f64)
            );
        }

        self.tokens = tokens.clone();
        let original_corpus = self
            .tokens_to_string(tokens)
            .chars()
            .take(400)
            .collect::<String>();
        println!("{original_corpus}");
    }
}

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    assert_eq!(args.len(), 2, "Expected only one argument!");

    let filepath = args[1].clone();

    println!("Read from file: {filepath}");

    let mut tokenizer = Tokenizer::new();

    let text = std::fs::read_to_string(filepath)?;

    tokenizer.train(text);

    Ok(())
}
