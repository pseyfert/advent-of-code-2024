pub mod binding;
// pub mod read_mod;
use crate::binding::pass;

struct Day20 {
    maze: Vec<i32>,
    goal_x: i32,
    goal_y: i32,
    rows: usize,
    cols: usize,
}

impl From<Vec<String>> for Day20 {
    fn from(input: Vec<String>) -> Day20 {
        let rows = input.len();
        let cols = input[0].len();

        let mut m = Vec::with_capacity(rows * cols);

        let mut goal_x = -1;
        let mut goal_y = -1;

        for (row_idx, row) in input.iter().enumerate() {
            for (col_idx, col) in row.chars().enumerate() {
                m.push(match col {
                    'E' => {
                        goal_x = col_idx as i32;
                        goal_y = row_idx as i32;
                        i32::MAX
                    }
                    'S' => 0,
                    '.' => i32::MAX,
                    '#' => -1,
                    _ => {
                        panic!("unexpected input");
                    }
                })
            }
        }
        assert!(goal_x >= 0);
        assert!(goal_y >= 0);
        Self {
            maze: m,
            goal_x,
            goal_y,
            rows,
            cols,
        }
    }
}

impl boiler_plate::Day for Day20 {
    type Desered = Vec<String>;

    fn part1(&self) -> anyhow::Result<u64> {
        let res = unsafe {
            pass(
                self.maze.as_ptr(),
                self.goal_x,
                self.goal_y,
                self.rows as i32,
                self.cols as i32,
            )
        };
        println!("result might be something like {res}");

        Ok(res as u64)
    }
}

boiler_plate::main!(Day20);
