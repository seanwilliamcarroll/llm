use super::token::Token;

pub trait Codec {
    fn encode(&self, input: &str) -> Vec<Token>;

    fn decode(&self, input: &[Token]) -> String;
}
