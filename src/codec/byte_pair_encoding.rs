use super::token::Token;
use super::types::Codec;
use std::collections::HashMap;

#[derive(Clone)]
pub struct BytePairEncodingCodec {
    encoding_rules: Vec<HashMap<Vec<u8>, Token>>,
    decoding_rules: HashMap<Token, Vec<u8>>,
}

impl BytePairEncodingCodec {
    pub fn new() -> Self {
        let rules = (0..=255u8).map(|value| (vec![value], Token::from_u8(value)));

        BytePairEncodingCodec {
            encoding_rules: vec![rules.clone().collect::<HashMap<Vec<u8>, Token>>()],
            decoding_rules: rules
                .map(|(k, v)| (v, k))
                .collect::<HashMap<Token, Vec<u8>>>(),
        }
    }

    fn add_encoding_rule(&mut self, new_token: Token, bytes: Vec<u8>) {
        if self.encoding_rules.len() < bytes.len() {
            self.encoding_rules.extend(vec![
                HashMap::new();
                bytes.len() - self.encoding_rules.len()
            ]);
        }
        self.encoding_rules[bytes.len() - 1].insert(bytes.clone(), new_token);
        self.decoding_rules.insert(new_token, bytes);
    }

    fn decode_token(&self, token: Token) -> Vec<u8> {
        self.decoding_rules
            .get(&token)
            .expect("Come back to this")
            .clone()
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

    fn frequency_count(tokens: &[Token]) -> Vec<(usize, Token)> {
        let mut counts: HashMap<Token, usize> = HashMap::new();

        for token in tokens {
            *counts.entry(*token).or_insert(0) += 1;
        }

        let mut output = counts.into_iter().map(|(k, v)| (v, k)).collect::<Vec<_>>();

        output.sort();
        output.reverse();
        output
    }

    pub fn print_vocab(&self, tokens: &[Token]) {
        let frequency_counts = Self::frequency_count(tokens);

        let mut most_impact_frequency = frequency_counts.clone();
        most_impact_frequency.sort_by_key(|(count, token)| {
            let bytes = self.decoding_rules.get(token).unwrap();
            std::cmp::Reverse(bytes.len() * count)
        });

        println!("Top 10 Tokens (Impact)");
        for (count, token) in most_impact_frequency.into_iter().take(10) {
            println!("--------------------------------------------------");
            let bytes = self.decoding_rules.get(&token).unwrap();
            print!(
                "{token} ({count} times with length {} = {} bytes)",
                bytes.len(),
                count * bytes.len()
            );
            if let Ok(string_token) = String::from_utf8(bytes.clone()) {
                println!("\n\"{string_token}\"");
            } else {
                println!();
            }
        }
        println!();
    }
}

impl Default for BytePairEncodingCodec {
    fn default() -> Self {
        Self::new()
    }
}

impl Codec for BytePairEncodingCodec {
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
                .flat_map(|token| self.decode_token(*token))
                .collect::<Vec<u8>>(),
        )
        .expect("Shouldn't fail")
    }
}

pub struct BytePairEncodingCodecTrainer {
    current_token_id: usize,
    mapping: HashMap<(Token, Token), Token>,
    reverse_mapping: HashMap<Token, (Token, Token)>,
    codec: BytePairEncodingCodec,
}

impl BytePairEncodingCodecTrainer {
    pub fn new() -> Self {
        Self {
            current_token_id: 256usize,
            mapping: HashMap::new(),
            reverse_mapping: HashMap::new(),
            codec: BytePairEncodingCodec::new(),
        }
    }

    pub fn get_codec(&self) -> BytePairEncodingCodec {
        self.codec.clone()
    }

    fn decode_token(&self, token: Token) -> Vec<u8> {
        // Need to build up from reverse mapping
        Self::tokens_to_bytes(self.expand_tokens(token))
            .expect("Do proper error checking here, but should just work")
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

    fn resize_with_additional_vocab(&mut self, additional_vocab: usize) {
        self.mapping.reserve(additional_vocab);
        self.reverse_mapping.reserve(additional_vocab);
    }

    fn add_encoding_rule(&mut self, pair: (Token, Token), new_token: Token) {
        assert_eq!(self.mapping.insert(pair, new_token), None);
        assert_eq!(self.reverse_mapping.insert(new_token, pair), None);

        let bytes = self.decode_token(new_token);
        assert!(bytes.len() > 1);
        self.codec.add_encoding_rule(new_token, bytes);
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

    pub fn train(&mut self, input: &str, additional_merges: usize) {
        let mut tokens = input.bytes().map(Token::from_u8).collect::<Vec<Token>>();
        let mut new_tokens: Vec<Token> = Vec::with_capacity(tokens.len());
        let mut current_tokens = Vec::with_capacity(2);

        self.resize_with_additional_vocab(additional_merges);

        for added_vocab in 0..additional_merges {
            if added_vocab == 2000 {
                // Sampling optimization, past this much vocab,
                // we can get away with just looking at first 200000 tokens
                tokens = tokens.into_iter().take(200_000).collect::<Vec<Token>>();
                new_tokens = Vec::with_capacity(tokens.len());
            }

            let pairs = Self::count_pairs(&tokens);

            let top_pairs = {
                let mut top_pairs = pairs.into_iter().collect::<Vec<_>>();
                top_pairs.sort_by_key(|(_, val)| std::cmp::Reverse(*val));
                top_pairs.into_iter().take(10).collect::<Vec<_>>()
            };

            let (top_pair, _) = top_pairs[0];

            tokens.reverse();

            while let Some(next_token) = tokens.pop() {
                current_tokens.push(next_token);
                match &current_tokens[..] {
                    [token_a, token_b] => {
                        let token_a = *token_a;
                        let token_b = *token_b;

                        if (token_a, token_b) == top_pair {
                            new_tokens.push(Token::from_usize(self.current_token_id));
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
            new_tokens.extend(current_tokens.drain(0..));
            current_tokens.clear();
            self.add_encoding_rule(top_pair, Token::from_usize(self.current_token_id));
            self.current_token_id += 1;
            std::mem::swap(&mut tokens, &mut new_tokens);
            new_tokens.clear();
        }
    }
}

impl Default for BytePairEncodingCodecTrainer {
    fn default() -> Self {
        Self::new()
    }
}
