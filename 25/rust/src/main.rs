#![feature(test)]

extern crate test;

use rayon::prelude::*;

struct Key {
    mask: [u8; 5],
}

struct Lock {
    mask: [u8; 5],
}

#[derive(thiserror::Error, Debug)]
#[error("error chopping input in elements")]
struct ReadError {}

boiler_plate::bench_parts!(Day25, "../input.txt");

impl TryFrom<Vec<String>> for Day25 {
    type Error = ReadError;

    fn try_from(v: Vec<String>) -> Result<Day25, Self::Error> {
        let mut it = v.iter();
        let mut l = Vec::new();
        let mut k = Vec::new();
        while let (Some(r1), Some(r2), Some(r3), Some(r4), Some(r5), Some(r6), Some(r7), _) = (
            it.next(),
            it.next(),
            it.next(),
            it.next(),
            it.next(),
            it.next(),
            it.next(),
            it.next(),
        ) {
            let a: [u8; 5] =
                itertools::izip!(r2.chars(), r3.chars(), r4.chars(), r5.chars(), r6.chars(),)
                    .map(|(c1, c2, c3, c4, c5)| {
                        [c1, c2, c3, c4, c5]
                            .into_iter()
                            .filter(|c| *c == '#')
                            .count() as u8
                    })
                    .collect::<Vec<u8>>()
                    .try_into()
                    .unwrap();

            if r1 == "#####" && r7 == "....." {
                l.push(Lock { mask: a });
            } else if r7 == "#####" && r1 == "....." {
                k.push(Key { mask: a });
            } else {
                return Err(Self::Error {});
            }
        }
        Ok(Day25 { locks: l, keys: k })
    }
}

struct Day25 {
    keys: Vec<Key>,
    locks: Vec<Lock>,
}

fn possible_match(k: &Key, l: &Lock) -> bool {
    k.mask
        .iter()
        .zip(l.mask.iter())
        .all(|(kn, ln)| kn + ln <= 5)
}

impl boiler_plate::Day for Day25 {
    type Desered = Vec<String>;

    fn part1(&self) -> anyhow::Result<u64> {
        Ok(self
            .locks
            .par_iter()
            .map(|l| {
                self.keys
                    .iter()
                    .filter(|k| possible_match(k, l))
                    .count() as u64
            })
            .sum::<u64>())
    }
}

boiler_plate::main!(Day25);
