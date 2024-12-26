#![feature(test)]

extern crate test;

use log::{debug, error};
use rayon::prelude::*;
use serde::{de, Deserialize};

boiler_plate::bench_parts!(Day14, "../input.txt");

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
mod unit_tests {
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

#[derive(Debug, Eq, PartialEq, Clone)]
struct Pos {
    x: i32,
    y: i32,
}

#[derive(Debug, Eq, PartialEq, Clone)]
struct Vel {
    x: i32,
    y: i32,
}

#[derive(Debug, Eq, PartialEq, Clone)]
struct Robot {
    initial_pos: Pos,
    vel: Vel,
}

#[derive(Debug, Eq, PartialEq, Clone, Hash)]
#[allow(non_camel_case_types)]
enum Quadrant {
    xy,
    xY,
    Xy,
    XY,
    Boarder,
}

impl From<Pos> for Quadrant {
    fn from(p: Pos) -> Self {
        match (p.x, p.y) {
            (0..=49, 0..=50) => Quadrant::xy,
            (0..=49, 52..=102) => Quadrant::xY,
            (51..=100, 0..=50) => Quadrant::Xy,
            (51..=100, 52..=102) => Quadrant::XY,
            (50, _) | (_, 51) => Quadrant::Boarder,
            _ => panic!(),
        }
    }
}

impl Robot {
    fn project(&self, steps: usize) -> Pos {
        fn trunk(p: i32, v: i32, steps: i32, dim: i32) -> i32 {
            let effective_num_steps = steps.rem_euclid(dim);
            let effective_move = (v * effective_num_steps).rem_euclid(dim);
            (p + effective_move).rem_euclid(dim)
        }

        Pos {
            x: trunk(self.initial_pos.x, self.vel.x, steps as i32, 101),
            y: trunk(self.initial_pos.y, self.vel.y, steps as i32, 103),
        }
    }

    #[allow(dead_code)]
    fn move_mut(&mut self, steps: usize) {
        self.initial_pos = self.project(steps);
    }
}

boiler_plate::just_wrap!(Day14, Vec<Robot>);

impl boiler_plate::Day for Day14 {
    type Desered = Vec<Robot>;
    // type Parsed = Self::Desered;

    fn part1(&self) -> anyhow::Result<u64> {
        // TODO: why?
        // let em = group_into::group_into::<_, _, _, _, enum_map::EnumMap<_, _>>(
        let em = group_into::group_into_hash_map(
            self.data
                .par_iter()
                .map(|r| -> Quadrant { r.project(100).into() })
                .filter(|q| *q != Quadrant::Boarder)
                .collect::<Vec<_>>() // TODO: compatibility issue
                // also: just count, no need to create the group
                .into_iter(),
            |q| q.clone(),
        );
        // group_by totally doesn't do what i think it does
        // .group_by(|q| q.clone())
        let part_one = em
            .into_iter() // TODO: no par_ here?
            .map(|(_key, group)| (_key, group.into_iter().count() as i128))
            .map(|(k, c)| {
                debug!("in quadrant {k:?} there are {c} robots");
                c
            })
            .product::<i128>();

        println!("part 1: {part_one}");
        error!("sorry, wrong int type");
        Ok(0)
    }

    fn part2(&self) -> anyhow::Result<u64> {
        let mut start = self.data.clone();

        for i in 0..=(101 * 103) {
            start.par_iter_mut().for_each(|r| r.move_mut(1));
            let em = group_into::group_into_hash_map(
                start
                    .par_iter()
                    .map(|r| (r.initial_pos.x - 50, r.initial_pos.y))
                    .collect::<Vec<_>>()
                    .into_iter(),
                |p| p.1,
            );
            if em.values().all(|xs| {
                let (negs, mut pos): (Vec<_>, Vec<_>) = xs
                    .into_iter()
                    .map(|p| p.0)
                    .filter(|x| *x != 0)
                    .partition(|x| *x < 0);
                let mut negs: Vec<_> = negs.into_iter().map(|x| -x).collect();
                pos.sort();
                negs.sort();
                negs == pos
            }) {
                println!("got a tree for {i}");
            }
        }

        Ok(0)
    }
}

boiler_plate::main!(Day14);
