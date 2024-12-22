use thiserror::Error;

use serde::de::{self, Deserialize, DeserializeSeed, IntoDeserializer, SeqAccess, Visitor};

#[derive(Error, Debug)]
#[error("something went wrong reading a file line by line: {some_message:?}")]
pub struct Error {
    some_message: String,
}

impl serde::de::Error for Error {
    fn custom<T>(msg: T) -> Self
    where
        T: core::fmt::Display,
    {
        Error {
            some_message: msg.to_string(),
        }
    }
}

impl Error {
    fn new(some_message: String) -> Self {
        Self { some_message }
    }
}

// largely copied from serde_plain

pub struct Deserializer<'de> {
    input: &'de str,
}

impl<'de> Deserializer<'de> {
    pub fn new(input: &'de str) -> Self {
        Deserializer { input }
    }

    // Look at the first character in the input without consuming it.
    fn peek_char(&mut self) -> Option<char> {
        self.input.chars().next()
    }
}

/// Deserialize an instance of type `T` from a string of plain text.
///
/// This deserializes the string into an object with the `Deserializer`
/// and returns it.  This requires that the type is a simple one
/// (integer, string etc.).
pub fn from_str<'a, T>(s: &'a str) -> Result<T, Error>
where
    T: Deserialize<'a>,
{
    let mut deserializer = Deserializer::new(s);
    T::deserialize(&mut deserializer)
}

macro_rules! forward_to_deserialize_from_str {
    ($func:ident, $visit_func:ident, $tymsg:expr) => {
        fn $func<V>(self, visitor: V) -> Result<V::Value, Error>
        where
            V: Visitor<'de>,
        {
            visitor.$visit_func(self.input.parse().map_err(|e| {
                Error::new(format!(
                    "parsing of {} as {} failed: {}",
                    self.input, $tymsg, e
                ))
            })?)
        }
    };
}

impl<'de> de::Deserializer<'de> for &mut Deserializer<'de> {
    type Error = Error;

    fn deserialize_any<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        self.deserialize_str(visitor)
    }

    forward_to_deserialize_from_str!(deserialize_bool, visit_bool, "boolean");
    forward_to_deserialize_from_str!(deserialize_i8, visit_i8, "i8");
    forward_to_deserialize_from_str!(deserialize_i16, visit_i16, "i16");
    forward_to_deserialize_from_str!(deserialize_i32, visit_i32, "i32");
    forward_to_deserialize_from_str!(deserialize_i64, visit_i64, "i64");
    forward_to_deserialize_from_str!(deserialize_u8, visit_u8, "u8");
    forward_to_deserialize_from_str!(deserialize_u16, visit_u16, "u16");
    forward_to_deserialize_from_str!(deserialize_u32, visit_u32, "u32");
    forward_to_deserialize_from_str!(deserialize_u64, visit_u64, "u64");
    forward_to_deserialize_from_str!(deserialize_f32, visit_f32, "f32");
    forward_to_deserialize_from_str!(deserialize_f64, visit_f64, "f64");
    forward_to_deserialize_from_str!(deserialize_char, visit_char, "char");

    fn deserialize_str<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        match self.input.find('\n') {
            Some(len) => {
                let rv = visitor.visit_borrowed_str(&self.input[..len])?;
                self.input = &self.input[len + 1..];
                Ok(rv)
            }
            None => visitor.visit_borrowed_str(self.input),
        }
    }

    fn deserialize_string<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        self.deserialize_str(visitor)
    }

    fn deserialize_bytes<V>(self, _visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        Err(Error::new("can't deserialize bytes".into()))
    }

    fn deserialize_byte_buf<V>(self, _visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        Err(Error::new("can't deserialize bytes".into()))
    }

    fn deserialize_option<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        if self.input.is_empty() {
            visitor.visit_none()
        } else {
            visitor.visit_some(self)
        }
    }

    fn deserialize_unit<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        if self.input.is_empty() {
            visitor.visit_unit()
        } else {
            Err(Error::new("expected empty string for unit".into()))
        }
    }

    fn deserialize_unit_struct<V>(self, _name: &'static str, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        self.deserialize_unit(visitor)
    }

    fn deserialize_newtype_struct<V>(
        self,
        _name: &'static str,
        visitor: V,
    ) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_newtype_struct(self)
    }

    fn deserialize_seq<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_seq(LineSeparation::new(self))
    }

    fn deserialize_tuple<V>(self, _len: usize, _visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        Err(Error::new("cant' do anything for tuple".into()))
    }

    fn deserialize_tuple_struct<V>(
        self,
        _name: &'static str,
        _len: usize,
        _visitor: V,
    ) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        Err(Error::new("cant' do anything for tuple struct".into()))
    }

    fn deserialize_map<V>(self, _visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        Err(Error::new("cant' do anything for map".into()))
    }

    fn deserialize_struct<V>(
        self,
        _name: &'static str,
        _fields: &'static [&'static str],
        _visitor: V,
    ) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        Err(Error::new("cant' do anything for struct".into()))
    }

    fn deserialize_enum<V>(
        self,
        _name: &'static str,
        _variants: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        visitor.visit_enum(self.input.into_deserializer())
    }

    fn deserialize_identifier<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        self.deserialize_str(visitor)
    }

    fn deserialize_ignored_any<V>(self, visitor: V) -> Result<V::Value, Error>
    where
        V: Visitor<'de>,
    {
        self.deserialize_any(visitor)
    }
}

struct LineSeparation<'a, 'de: 'a> {
    de: &'a mut Deserializer<'de>,
}

impl<'a, 'de> LineSeparation<'a, 'de> {
    fn new(de: &'a mut Deserializer<'de>) -> Self {
        LineSeparation { de }
    }
}

// `SeqAccess` is provided to the `Visitor` to give it the ability to iterate
// through elements of the sequence.
impl<'de> SeqAccess<'de> for LineSeparation<'_, 'de> {
    type Error = Error;

    fn next_element_seed<T>(&mut self, seed: T) -> Result<Option<T::Value>, Error>
    where
        T: DeserializeSeed<'de>,
    {
        if self.de.peek_char() == None {
            return Ok(None);
        }
        // Deserialize an array element.
        seed.deserialize(&mut *self.de).map(Some)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_number() {
        let v: Vec<u32> = from_str("123\n456\n").unwrap();
        assert_eq!(v[0], 123_u32);
        assert_eq!(v[1], 456_u32);
    }
    #[test]
    fn test_something() {
        let v: Vec<Word> = from_str("foo\nbar\n").unwrap();
        assert_eq!(v[0].w, "foo");
        assert_eq!(v[1].w, "bar");
    }

    struct Word {
        w: String,
    }

    impl<'de> Deserialize<'de> for Word {
        fn deserialize<D>(deserializer: D) -> Result<Word, D::Error>
        where
            D: de::Deserializer<'de>,
        {
            let s: &str = de::Deserialize::deserialize(deserializer)?;

            Ok(Word { w: s.to_string() })
        }
    }
}
