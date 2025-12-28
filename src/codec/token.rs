use std::convert::TryFrom;

#[derive(Debug, Hash, Eq, PartialEq, Clone, Copy, PartialOrd, Ord)]
pub struct Token(usize);

impl Token {
    pub fn as_byte(self) -> Option<u8> {
        u8::try_from(self.0).ok()
    }

    pub fn from_usize(input: usize) -> Self {
        Self(input)
    }

    pub fn from_u8(input: u8) -> Self {
        Self(input as usize)
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
