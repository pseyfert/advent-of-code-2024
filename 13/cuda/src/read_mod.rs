use serde::{de, Deserialize};

impl<'de> Deserialize<'de> for Machine {
    fn deserialize<D>(deserializer: D) -> Result<Machine, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        let s: &str = de::Deserialize::deserialize(deserializer)?;
        let words = s.split(' ').collect::<Vec<&str>>();
        if words.len() != 11 {
            return Err(de::Error::custom(format!(
                "Incorrect number of elements per row",
            )));
        }

        fn validate<E: serde::de::Error>(lhs: &str, rhs: &str) -> Result<(), E> {
            if lhs != rhs {
                Err(E::custom(format!(
                    "expected '{}' instead of '{}'.",
                    lhs, rhs
                )))
            } else {
                Ok(())
            }
        }

        validate::<D::Error>(words[0], "Button")?;
        validate::<D::Error>(words[1], "A:")?;
        validate::<D::Error>(words[4], "Button")?;
        validate::<D::Error>(words[5], "B:")?;
        validate::<D::Error>(words[8], "Prize:")?;

        Ok(Machine {
            Ax: words[2]
                .strip_prefix("X+")
                .ok_or(de::Error::custom("missing prefix"))?
                .strip_suffix(",")
                .ok_or(de::Error::custom("missing comma in button A"))?
                .parse()
                .map_err(de::Error::custom)?,
            Ay: words[3]
                .strip_prefix("Y+")
                .ok_or(de::Error::custom("missing prefix"))?
                .parse()
                .map_err(de::Error::custom)?,
            Bx: words[6]
                .strip_prefix("X+")
                .ok_or(de::Error::custom("missing prefix"))?
                .strip_suffix(",")
                .ok_or(de::Error::custom("missing comma in button B"))?
                .parse()
                .map_err(de::Error::custom)?,
            By: words[7]
                .strip_prefix("Y+")
                .ok_or(de::Error::custom("missing prefix"))?
                .parse()
                .map_err(de::Error::custom)?,
            Tx: words[9]
                .strip_prefix("X=")
                .ok_or(de::Error::custom("missing prefix"))?
                .strip_suffix(",")
                .ok_or(de::Error::custom("missing comma in prize"))?
                .parse()
                .map_err(de::Error::custom)?,
            Ty: words[10]
                .strip_prefix("Y=")
                .ok_or(de::Error::custom("missing prefix"))?
                .parse()
                .map_err(de::Error::custom)?,
        })
    }
}

pub struct Machine {
    pub Ax: i32,
    pub Ay: i32,
    pub Bx: i32,
    pub By: i32,
    pub Tx: i32,
    pub Ty: i32,
}
