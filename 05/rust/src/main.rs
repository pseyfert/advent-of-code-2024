use itertools::Itertools;
use rayon::prelude::*;
use serde::{de, Deserialize};
use std::path::Path;
use log::debug;

boiler_plate::main!(Day05);
impl<'de> Deserialize<'de> for Day05 {
    fn deserialize<D>(_deserializer: D) -> Result<Day05, D::Error>
    where
        D: de::Deserializer<'de>,
    {
        todo!("should not be used");
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
        Ok(Rule {
            lhs: lhs.parse().map_err(de::Error::custom)?,
            rhs: rhs.parse().map_err(de::Error::custom)?,
            order: Order::LeftThenRight,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Order {
    LeftThenRight,
    RightThenLeft,
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
    pub order: Order,
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

        let mut rules = rules.to_string() + "\n";
        debug!("updates is {updates:?}");
        let mut rules: Vec<Rule> = serde_linewise::from_str(&rules)?;
        let updates: Vec<Update> = serde_linewise::from_str(updates)?;

        let mut mirror_rules = rules
            .iter()
            .map(|r| Rule {
                lhs: r.rhs,
                rhs: r.lhs,
                order: Order::RightThenLeft,
            })
            .collect::<Vec<_>>();

        rules.append(&mut mirror_rules);
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
}

fn check_validity(u: &Update, r: &[Rule]) -> bool {
    itertools::all(u.pages.iter().combinations(2), |comb| {
        let mut it = comb.iter();
        let (lhs, rhs) = (it.next().unwrap(), it.next().unwrap());
        debug!("checking order of {lhs} and {rhs}");
        let first_equal_or_too_far_idx = r.partition_point(|rule| rule.lhs < **lhs);
        r[first_equal_or_too_far_idx..]
            .iter()
            .inspect(|rule| debug!("there is rule {rule:?}"))
            .take_while(|rule| rule.lhs == **lhs)
            .filter(|rule| rule.rhs == **rhs)
            .all(|rule| {
                debug!("considering rule {rule:?}");
                match rule.order {
                    Order::LeftThenRight => true,
                    // Note to self: why do i even keep the other part of the list
                    Order::RightThenLeft => false,
                }
            })
    })
}
