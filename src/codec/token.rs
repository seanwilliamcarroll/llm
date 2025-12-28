use serde::{Deserialize, Serialize};
use std::convert::TryFrom;

pub type TokenInternal = u32;

#[derive(Debug, Hash, Eq, PartialEq, Clone, Copy, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Token(TokenInternal);

impl Token {
    pub fn as_byte(self) -> Option<u8> {
        u8::try_from(self.0).ok()
    }
}

impl From<u32> for Token {
    fn from(input: u32) -> Self {
        Token(TokenInternal::from(input))
    }
}

impl From<u8> for Token {
    fn from(input: u8) -> Self {
        Token(TokenInternal::from(input))
    }
}

impl std::fmt::Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if self.0 < TokenInternal::from(256u16) {
            let character = char::from_u32(self.0 as TokenInternal).unwrap();
            if character.is_ascii() {
                return write!(f, "T<{}({:?})>", self.0, character);
            }
        }
        write!(f, "T<{}>", self.0)
    }
}
