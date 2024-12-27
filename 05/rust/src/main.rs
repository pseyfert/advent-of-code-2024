#![feature(test)]
extern crate test;
use itertools::Itertools;
use log::debug;
use rayon::prelude::*;
use serde::{de, Deserialize};
use std::path::Path;

boiler_plate::bench_parts!(Day05, "../input.txt");
boiler_plate::main!(Day05);
impl<'de> Deserialize<'de> for Day05 {
    fn deserialize<D>(_deserializer: D) -> Result<Day05, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        todo!("should not be used");
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    use boiler_plate::Day;
    #[test]
    fn p1() {
        let input = Day05::deser("../example.txt").unwrap().try_into().unwrap();
        assert_eq!(Day05::part1(&input).unwrap(), 143);
    }
}

impl<'de> Deserialize<'de> for Update {
    fn deserialize<D>(deserializer: D) -> Result<Update, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        let s: &str = de::Deserialize::deserialize(deserializer)?;
        let pages = s
            .split(',')
            .map(|s| s.parse::<u16>().map_err(de::Error::custom))
            .collect::<Result<Vec<u16>, D::Error>>()?;
        Ok(Update { pages })
    }
}

impl<'de> Deserialize<'de> for Rule {
    fn deserialize<D>(deserializer: D) -> Result<Rule, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        let s: &str = de::Deserialize::deserialize(deserializer)?;
        let mut part = s.split('|');
        let (Some(lhs), Some(rhs), None) = (part.next(), part.next(), part.next()) else {
            return Err(de::Error::custom(format!("invalid line: {}.", s)));
        };
        // NB: late refactor, they swap.
        Ok(Rule {
            rhs: lhs.parse().map_err(de::Error::custom)?,
            lhs: rhs.parse().map_err(de::Error::custom)?,
        })
    }
}

impl PartialOrd for Rule {
    fn partial_cmp(&self, other: &Rule) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Rule {
    fn cmp(&self, other: &Rule) -> std::cmp::Ordering {
        match self.lhs.cmp(&other.lhs) {
            std::cmp::Ordering::Equal => self.rhs.cmp(&other.rhs),
            o => o,
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct Rule {
    pub lhs: u16,
    pub rhs: u16,
}

#[derive(Debug, Clone)]
struct Update {
    pub pages: Vec<u16>,
}

struct Day05 {
    rules: Vec<Rule>,
    updates: Vec<Update>,
}

impl boiler_plate::Day for Day05 {
    type Desered = Self;

    fn deser(p: impl AsRef<Path>) -> anyhow::Result<Self> {
        let bare_input = std::fs::read_to_string(p)?;
        let mut bare_input_split = bare_input.split("\n\n");
        let (Some(rules), Some(updates), None) = (
            bare_input_split.next(),
            bare_input_split.next(),
            bare_input_split.next(),
        ) else {
            anyhow::bail!("Could not find the empty line");
        };

        let rules = rules.to_string() + "\n";
        debug!("updates is {updates:?}");
        let mut rules: Vec<Rule> = serde_linewise::from_str(&rules)?;
        let updates: Vec<Update> = serde_linewise::from_str(updates)?;

        rules.par_sort();
        Ok(Self { rules, updates })
    }

    fn part1(&self) -> anyhow::Result<u64> {
        Ok(self
            .updates
            .par_iter()
            .filter_map(|u| -> Option<u64> {
                if check_validity(u, &self.rules) {
                    Some(u.pages[u.pages.len() / 2] as u64)
                } else {
                    None
                }
            })
            .sum())
    }
    fn part2(&self) -> anyhow::Result<u64> {
        let incorrect_ones = self.prepare_part2();
        Ok(incorrect_ones
            .into_par_iter()
            .map(|mut u| {
                u.pages
                    .sort_unstable_by(|page1, page2| rule_compare(page1, page2, &self.rules));
                u.pages[u.pages.len() / 2] as u64
            })
            .sum())
    }
}

fn rule_compare(p1: &u16, p2: &u16, rules: &[Rule]) -> std::cmp::Ordering {
    let first_equal_or_too_far_idx = rules.partition_point(|rule| rule.lhs < *p1);
    let Some(_) = rules[first_equal_or_too_far_idx..]
        .iter()
        .take_while(|rule| rule.lhs == *p1)
        .find(|rule| rule.rhs == *p2)
    else {
        return std::cmp::Ordering::Less;
    };
    std::cmp::Ordering::Greater
}

fn check_validity(u: &Update, r: &[Rule]) -> bool {
    itertools::all(u.pages.iter().combinations(2), |comb| {
        let mut it = comb.iter();
        let (lhs, rhs) = (it.next().unwrap(), it.next().unwrap());
        debug!("checking order of {lhs} and {rhs}");
        let first_equal_or_too_far_idx = r.partition_point(|rule| rule.lhs < **lhs);
        r[first_equal_or_too_far_idx..]
            .iter()
            // .inspect(|rule| debug!("there is rule {rule:?}"))
            .take_while(|rule| rule.lhs == **lhs)
            .filter(|rule| rule.rhs == **rhs)
            .next()
            .is_none()
    })
}

impl Day05 {
    fn prepare_part2(&self) -> Vec<Update> {
        self.updates
            .par_iter()
            .filter_map(|u| {
                if !check_validity(u, &self.rules) {
                    Some(u.clone())
                } else {
                    None
                }
            })
            .collect()
    }
}
