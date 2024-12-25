use log::{debug, trace};

struct Day20 {}

#[derive(Debug, Hash, Eq, PartialEq, Clone)]
enum CheatStatus {
    Uncheated,
    MidCheat,
    Cheated,
}

impl boiler_plate::Day for Day20 {
    fn parse(desered: Self::Desered) -> anyhow::Result<Self::Parsed> {
        trace!("{desered:?}");
        Ok(desered.into())
    }

    type Desered = Vec<String>;
    type Parsed = grid::Maze;

    fn process(g: grid::Maze) -> anyhow::Result<()> {
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

        let mut cost_lookup = mdarray::DGrid::<usize, 2>::new();
        cost_lookup.resize([g.grid.dim_x, g.grid.dim_y], 0);

        for (dist_from_start, point) in initial_path.iter().enumerate() {
            cost_lookup[[point.x, point.y]] = dist_from_start;
        }
        let part1 = initial_path
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
                    .filter_map(|o| o)
                    .filter(|p| cost_lookup[[p.x, p.y]] >= c + 102)
                    .count()
            })
            .sum::<usize>();

        println!("there should be about {part1} shortcuts");

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

        Ok(())
    }
}

fn main() -> std::process::ExitCode {
    boiler_plate::main_wrap::<Day20>()
}
