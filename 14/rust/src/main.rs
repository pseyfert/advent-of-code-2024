use serde::{de, Deserialize};

impl<'de> Deserialize<'de> for Robot {
    fn deserialize<D>(deserializer: D) -> Result<Robot, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        let s: &str = de::Deserialize::deserialize(deserializer)?;
        let mut part = s.split(' ');
        let (Some(p), Some(v), None) = (part.next(), part.next(), part.next()) else {
            return Err(de::Error::custom(format!("invalid line: {}.", s)));
        };
        let p = p
            .strip_prefix("p=")
            .ok_or(de::Error::custom("missing p= prefix"))?;
        let v = v
            .strip_prefix("v=")
            .ok_or(de::Error::custom("missing v= prefix"))?;

        fn two_d<'de, D>(s: &str) -> Result<(i32, i32), D::Error>
        where
            D: de::Deserializer<'de>,
        {
            let mut i = s.split(',');
            let (Some(x), Some(y), None) = (i.next(), i.next(), i.next()) else {
                return Err(de::Error::custom(format!("coordinates: {}.", s)));
            };
            Ok((
                x.parse::<i32>().map_err(de::Error::custom)?,
                y.parse::<i32>().map_err(de::Error::custom)?,
            ))
        }

        let (px, py) = two_d::<D>(p)?;
        let (vx, vy) = two_d::<D>(v)?;
        Ok(Robot {
            initial_pos: Pos { x: px, y: py },
            vel: Vel { x: vx, y: vy },
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use rstest::rstest;

    #[rstest]
    #[case("p=92,72 v=-49,-72", Robot{initial_pos: Pos{x:92, y:72}, vel: Vel{x:-49, y:-72}})]
    #[case("p=0,53 v=27,-81", Robot{initial_pos: Pos{x:0, y:53}, vel: Vel{x:27, y:-81}})]
    fn test_robot_deser(#[case] s: &str, #[case] robot: Robot) {
        assert_eq!(serde_plain::from_str::<Robot>(s).unwrap(), robot);
    }

    #[test]
    fn test_example() {
        let v: Vec<Robot> =
            serde_linewise::from_str(&std::fs::read_to_string("../example.txt").unwrap()).unwrap();
        assert_eq!(
            v[2],
            Robot {
                initial_pos: Pos { x: 10, y: 3 },
                vel: Vel { x: -1, y: 2 }
            }
        );
    }
}

#[derive(Debug, Eq, PartialEq)]
struct Pos {
    x: i32,
    y: i32,
}

impl Pos {
    fn normalize(&self) {
        todo!();
    }
}

#[derive(Debug, Eq, PartialEq)]
struct Vel {
    x: i32,
    y: i32,
}

#[derive(Debug, Eq, PartialEq)]
struct Robot {
    initial_pos: Pos,
    vel: Vel,
}

fn main() {}
