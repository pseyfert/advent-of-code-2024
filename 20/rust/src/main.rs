#![feature(test)]

extern crate test;

use log::{debug, trace};
use rayon::prelude::*;

boiler_plate::just_wrap!(Day20, grid::Maze);
boiler_plate::bench_parts!(Day20, "../input.txt");

#[derive(Debug, Hash, Eq, PartialEq, Clone)]
enum CheatStatus {
    Uncheated,
    MidCheat,
    Cheated,
}

impl boiler_plate::Day for Day20 {
    fn part2(&self) -> anyhow::Result<u64> {
        let (initial_path, base_cost) = Self::initial(&self.data)?;

        let potential_starts = &initial_path[0..(base_cost - 100) as usize];
        Ok(potential_starts
            .par_iter()
            .enumerate()
            // take_while does not work with par_iter, thus switching to potential_starts
            // .take_while(|(i, _)| (*i as u32) < base_cost - 100)
            // .filter(|(i, _)| (*i as u32) < base_cost - 100)
            .map(|(i, p1)| {
                let mut counter = 0;
                let mut iter = initial_path.iter().enumerate().skip(i + 101);
                while let Some((j, p2)) = iter.next() {
                    let cheat_length = grid::manhattan(p1, p2);
                    if cheat_length > 20 {
                        // tweak
                        // For cheat length N = 20 + s we need to go at least s steps further in
                        // the path
                        for _ in 0..cheat_length - 21 {
                            iter.next();
                        }
                        continue;
                    }
                    // if i < 1 && base_cost - 50 < j as u32 {
                    //     debug!(
                    //         "testing a cheat {i}: {p1:?} → {j}: {p2:?} of length {cheat_length}"
                    //     );
                    // }
                    let (gain, allowed) = (j - i).overflowing_sub(cheat_length);
                    if !allowed && (gain >= 100) {
                        counter += 1;
                    }
                }
                counter
            })
            .sum::<u64>())
    }

    fn part1(&self) -> anyhow::Result<u64> {
        let (initial_path, base_cost) = Self::initial(&self.data)?;

        let cost_lookup = Self::cost_lookup(&self.data, &initial_path);

        Ok(initial_path
            .iter()
            .enumerate()
            .take_while(|(i, _)| (*i as u32) < base_cost - 100)
            .map(|(c, p)| {
                let nn = p.north().and_then(|p| p.north());
                let nw = p.north().and_then(|p| p.west());
                let ww = p.west().and_then(|p| p.west());
                let sw = p.west().and_then(|p| p.south());
                let ss = p.south().and_then(|p| p.south());
                let se = p.east().and_then(|p| p.south());
                let ee = p.east().and_then(|p| p.east());
                let ne = p.north().and_then(|p| p.east());
                [nn, nw, ww, sw, ss, se, ee, ne]
                    .into_iter()
                    .flatten()
                    .filter(|p| cost_lookup[[p.x, p.y]] >= c + 102)
                    .count() as u64
            })
            .sum::<u64>())

        // // TODO: why does astar_bag blow up?
        // let mut hmmm = std::collections::HashMap::new();
        //
        // for (dist_from_start, point) in initial_path.iter().enumerate() {
        //     hmmm.insert(point, base_cost - dist_from_start as u32);
        // }
        // let Some(cheats) =
        //     pathfinding::directed::astar:://astar(
        //         astar_bag_collect(
        //         &(start.clone(), CheatStatus::Uncheated),
        //         |(p, stat)| -> Vec<((grid::GridPoint<'_>, CheatStatus), u32)> {
        //             if *p == start {
        //                 g.grid
        //                     .orth_neighbours(p)
        //                     .filter_map(|p| match g.at(&p) {
        //                         grid::MazePoint::Wall => Some(((p, CheatStatus::MidCheat), 1)),
        //                         _ => None,
        //                     })
        //                     .chain(initial_path.iter().enumerate().map(|(cost, p)| {
        //                         ((p.clone(), CheatStatus::Uncheated), cost as u32)
        //                     }))
        //                     .collect::<Vec<_>>()
        //             } else {
        //                 match stat {
        //                     CheatStatus::Cheated => {
        //                         vec![((end.clone(), CheatStatus::Cheated), *hmmm.get(p).unwrap())]
        //                     }
        //                     CheatStatus::MidCheat => g
        //                         .unblocked_orth_neighbours(p)
        //                         .into_iter()
        //                         .map(|p| ((p, CheatStatus::Cheated), 1))
        //                         .collect::<Vec<_>>(),
        //                     CheatStatus::Uncheated => g
        //                         .grid
        //                         .orth_neighbours(p)
        //                         .filter_map(|p| match g.at(&p) {
        //                             grid::MazePoint::Wall => Some(((p, CheatStatus::MidCheat), 1)),
        //                             _ => None,
        //                         })
        //                         .collect::<Vec<_>>(),
        //                 }
        //             }
        //         },
        //         |(p, stat)| match stat {
        //             CheatStatus::Uncheated => grid::manhattan(p, &end) as u32,
        //             CheatStatus::MidCheat => grid::manhattan(p, &end) as u32,
        //             CheatStatus::Cheated => match hmmm.get(p) {
        //                 Some(cost) => *cost,
        //                 None => {
        //                     panic!()
        //                 }
        //             },
        //         },
        //         |(p, _stat)| g.at(p) == grid::MazePoint::End,
        //     )
        //     .ok_or(anyhow::anyhow!("couldn't find the path"))
        //     .into_iter()
        //     .next()
        // else {
        //     return Err(anyhow::anyhow!("somehow could but couldn't find the path"));
        // };
        //
        // debug!("{cheats:?}");
    }

    type Desered = Vec<String>;
}

impl Day20 {
    fn initial(g: &grid::Maze) -> anyhow::Result<(Vec<grid::GridPoint<'_>>, u32)> {
        let start = g
            .start()
            .ok_or(anyhow::anyhow!("didn't find start in maze"))?;
        let end = g.end().ok_or(anyhow::anyhow!("didn't find end in maze"))?;
        debug!("calling astar");
        let Some((initial_path, base_cost)) = pathfinding::directed::astar::astar(
            &start,
            |p| {
                g.unblocked_orth_neighbours(p)
                    .into_iter()
                    .inspect(|p| {
                        trace!("suggesting {p:?} to go next");
                    })
                    .map(|p| (p, 1u32))
            },
            |p| grid::manhattan(p, &end) as u32,
            |p| g.at(p) == grid::MazePoint::End,
        )
        .ok_or(anyhow::anyhow!("couldn't find the path"))
        .into_iter()
        .next() else {
            return Err(anyhow::anyhow!("somehow could but couldn't find the path"));
        };
        Ok((initial_path, base_cost))
    }

    fn cost_lookup<'a>(
        g: &'a grid::Maze,
        initial_path: &Vec<grid::GridPoint<'a>>,
    ) -> mdarray::DGrid<usize, 2> {
        let mut cost_lookup = mdarray::DGrid::<usize, 2>::new();
        cost_lookup.resize([g.grid.dim_x, g.grid.dim_y], 0);

        for (dist_from_start, point) in initial_path.iter().enumerate() {
            cost_lookup[[point.x, point.y]] = dist_from_start;
        }

        cost_lookup
    }
}

fn main() -> std::process::ExitCode {
    boiler_plate::main_wrap::<Day20>()
}
