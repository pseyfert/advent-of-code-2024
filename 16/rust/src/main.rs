boiler_plate::main!(Day16);

boiler_plate::just_wrap!(Day16, grid::Maze);

impl boiler_plate::Day for Day16 {
    type Desered = Vec<String>;

    fn part1(&self) -> anyhow::Result<u64> {
        let start = self.data.start().ok_or(anyhow::anyhow!("no start?"))?;
        let end = self.data.end().ok_or(anyhow::anyhow!("no end?"))?;

        let Some((_best_path, cost)) = pathfinding::directed::astar::astar_bag(
            &(start, grid::Direction::East),
            |(pos, dir)| {
                [
                    ((pos.clone(), dir.cw()), 1000),
                    ((pos.clone(), dir.ccw()), 1000),
                ]
                .into_iter()
                .chain(
                    if let Some(forward) = pos
                        .direction(dir.clone())
                        .map(|p| {
                            if self.data.at(&p) == grid::MazePoint::Wall {
                                None
                            } else {
                                Some(p)
                            }
                        })
                        .flatten()
                    {
                        vec![((forward, dir.clone()), 1)]
                    } else {
                        vec![]
                    },
                )
            },
            |(pos, dir)| heuristic(pos, dir, &end),
            |(pos, _)| *pos == end,
        ) else {
            anyhow::bail!("didn't manage to solve maze");
        };
        Ok(cost as u64)
    }
}

fn heuristic<'a>(
    pos: &grid::GridPoint<'a>,
    dir: &grid::Direction,
    end: &grid::GridPoint<'a>,
) -> usize {
    grid::manhattan(pos, end)
        // TODO: there's gotta be a more "clever" way to do this
        + if pos.x == end.x {
            match dir {
                grid::Direction::East | grid::Direction::West => 1000,
                grid::Direction::North => {
                    if pos.y > end.y {
                        0
                    } else {
                        2000
                    }
                }
                grid::Direction::South => {
                    if pos.y > end.y {
                        2000
                    } else {
                        0
                    }
                }
            }
        } else if pos.y == end.y {
            match dir {
                grid::Direction::North | grid::Direction::South => 1000,
                grid::Direction::East => {
                    if pos.x > end.x {
                        0
                    } else {
                        2000
                    }
                }
                grid::Direction::West => {
                    if pos.x > end.x {
                        2000
                    } else {
                        0
                    }
                }
            }
        } else if pos.y > end.y && pos.x > end.x {
            match dir {
                grid::Direction::North => 1000,
                grid::Direction::South => 2000,
                grid::Direction::East => 2000,
                grid::Direction::West => 1000,
            }
        } else if pos.y > end.y && pos.x < end.x {
            match dir {
                grid::Direction::North => 1000,
                grid::Direction::South => 2000,
                grid::Direction::East => 1000,
                grid::Direction::West => 2000,
            }
        } else if pos.y < end.y && pos.x > end.x {
            match dir {
                grid::Direction::North => 2000,
                grid::Direction::South => 1000,
                grid::Direction::East => 2000,
                grid::Direction::West => 1000,
            }
        } else if pos.y < end.y && pos.x < end.x {
            match dir {
                grid::Direction::North => 2000,
                grid::Direction::South => 1000,
                grid::Direction::East => 1000,
                grid::Direction::West => 2000,
            }
        } else {
            panic!()
        }
}
