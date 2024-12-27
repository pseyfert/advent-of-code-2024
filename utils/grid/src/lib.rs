use log::debug;

#[derive(Eq, PartialEq, Debug, Clone)]
pub enum MazePoint {
    Wall,
    Free,
    Start,
    End,
}

// #[derive(thiserror::Error, Debug)]
// #[error("Unexpected element in maze string")]
// pub struct FromStringToMazeError {}

// TODO: would prefer TryFrom, but didn't get boiler_plate working quickly
// impl TryFrom<char> for MazePoint {
//     type Error = FromStringToMazeError;
//     fn try_from(c: char) -> Result<MazePoint, FromStringToMazeError> {
impl From<char> for MazePoint {
    fn from(c: char) -> MazePoint {
        match c {
            '.' => MazePoint::Free,
            'S' => MazePoint::Start,
            'E' => MazePoint::End,
            '#' => MazePoint::Wall,
            _ => panic!(),
        }
    }
}

// impl TryFrom<&Vec<String>> for Maze {
//     type Error = FromStringToMazeError;
//     fn try_from(v: &Vec<String>) -> Result<Maze, FromStringToMazeError> {
impl From<Vec<String>> for Maze {
    fn from(v: Vec<String>) -> Maze {
        (&v).into()
    }
}
impl From<&Vec<String>> for Maze {
    fn from(v: &Vec<String>) -> Maze {
        debug!("should see this");
        assert!(v.len() > 0);
        let g = Grid {
            dim_x: v[0].len(),
            dim_y: v.len(),
        };
        debug!("Setting up a grid: {g:?}");
        let mut m = mdarray::DGrid::<MazePoint, 2>::new();
        let mut x = None;
        let mut y = None;
        let mut xe = None;
        let mut ye = None;
        m.resize([g.dim_x, g.dim_y], MazePoint::Wall);
        for (row_idx, row) in v.iter().enumerate() {
            assert_eq!(row.len(), g.dim_x);
            for (col_idx, col) in row.chars().enumerate() {
                // m[[col_idx, row_idx]] = col.try_into()?;
                let here: MazePoint = col.into();
                m[[col_idx, row_idx]] = here.clone();
                if here == MazePoint::Start {
                    x = Some(col_idx);
                    y = Some(row_idx);
                }
                if here == MazePoint::End {
                    xe = Some(col_idx);
                    ye = Some(row_idx);
                }
            }
        }
        Maze {
            grid: g,
            data: m,
            start_x: x,
            start_y: y,
            end_x: xe,
            end_y: ye,
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_constructor() {
        let v = vec!["####".to_string(), "#S.E".to_string()];
        // let m = Maze::try_from(&v).unwrap();
        let m = Maze::try_from(&v).unwrap();
        assert_eq!(
            m.at(&GridPoint {
                grid: &m.grid,
                x: 1,
                y: 1,
            }),
            MazePoint::Start
        );
        assert_eq!(
            m.at(&GridPoint {
                grid: &m.grid,
                x: 3,
                y: 1,
            }),
            MazePoint::End
        );
    }
}

pub struct Maze {
    pub grid: Grid, // TODO: technically implied by data …
    // NB: mdarray is column-major
    pub data: mdarray::DGrid<MazePoint, 2>,
    start_x: Option<usize>,
    start_y: Option<usize>,
    end_x: Option<usize>,
    end_y: Option<usize>,
}

impl Maze {
    pub fn end(&self) -> Option<GridPoint> {
        Some(GridPoint {
            grid: &self.grid,
            x: self.end_x?,
            y: self.end_y?,
        })
    }
    pub fn start(&self) -> Option<GridPoint> {
        Some(GridPoint {
            grid: &self.grid,
            x: self.start_x?,
            y: self.start_y?,
        })
    }
    pub fn at(&self, coords: &GridPoint) -> MazePoint {
        self.data[[coords.x, coords.y]].clone()
    }
    pub fn filter_walls<'a>(
        &self,
        it: impl Iterator<Item = GridPoint<'a>>,
    ) -> impl Iterator<Item = GridPoint<'a>> {
        it.filter(|gp| self.at(gp) != MazePoint::Wall)
    }
    pub fn unblocked_orth_neighbours<'a>(&self, center: &GridPoint<'a>) -> Vec<GridPoint<'a>> {
        self.filter_walls(self.grid.orth_neighbours(center))
            .collect()
    }
}

impl Grid {
    pub fn orth_neighbours<'a>(
        &self,
        center: &GridPoint<'a>,
    ) -> impl Iterator<Item = GridPoint<'a>> {
        enum_iterator::all::<Direction>().filter_map(move |d| center.direction(d))
    }
}

#[derive(Eq, PartialEq, Debug, Clone, Hash)]
pub struct Grid {
    pub dim_x: usize,
    pub dim_y: usize,
}

// TODO: get rid of Hash
#[derive(Eq, PartialEq, Debug, Clone, Hash)]
pub struct GridPoint<'a> {
    grid: &'a Grid,
    // TODO: better allow using custom mdarray
    pub x: usize,
    pub y: usize,
}

#[derive(Eq, PartialEq, Debug, Clone, enum_iterator::Sequence, Hash)]
pub enum Direction {
    North,
    South,
    East,
    West,
}

impl Direction {
    pub fn ccw(&self) -> Self {
        match self {
            Self::North => Self::West,
            Self::South => Self::East,
            Self::East => Self::North,
            Self::West => Self::South,
        }
    }
    pub fn cw(&self) -> Self {
        match self {
            Self::North => Self::East,
            Self::South => Self::West,
            Self::East => Self::South,
            Self::West => Self::North,
        }
    }
}

pub fn manhattan(lhs: &GridPoint, rhs: &GridPoint) -> usize {
    lhs.x.abs_diff(rhs.x) + lhs.y.abs_diff(rhs.y)
}

impl GridPoint<'_> {
    pub fn direction(&self, dir: Direction) -> Option<Self> {
        match dir {
            Direction::North => self.north(),
            Direction::South => self.south(),
            Direction::East => self.east(),
            Direction::West => self.west(),
        }
    }
    pub fn north(&self) -> Option<Self> {
        match self.y.overflowing_sub(1) {
            (y, false) => Some(Self {
                grid: self.grid,
                x: self.x,
                y,
            }),
            (_, true) => None,
        }
    }
    pub fn south(&self) -> Option<Self> {
        let y = self.y + 1;
        if y < self.grid.dim_y {
            Some(Self {
                grid: self.grid,
                x: self.x,
                y,
            })
        } else {
            None
        }
    }
    pub fn east(&self) -> Option<Self> {
        let x = self.x + 1;
        if x < self.grid.dim_x {
            Some(Self {
                grid: self.grid,
                x,
                y: self.y,
            })
        } else {
            None
        }
    }
    pub fn west(&self) -> Option<Self> {
        match self.x.overflowing_sub(1) {
            (x, false) => Some(Self {
                grid: self.grid,
                x,
                y: self.y,
            }),
            (_, true) => None,
        }
    }
}
