pub mod binding;
// pub mod read_mod;
use crate::binding::pass;

struct Data {
    maze: Vec<i32>,
    goal_x: i32,
    goal_y: i32,
    rows: usize,
    cols: usize,
}

impl From<Vec<String>> for Data {
    fn from(_: Vec<String>) -> Data {
        // TODO: this is silly. Overriding parse is pointless.
        panic!();
    }
}

impl boiler_plate::Day for Day20 {
    type Desered = Vec<String>;
    type Parsed = Data;

    fn parse(input: Self::Desered) -> anyhow::Result<Self::Parsed> {
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
        Ok(Data {
            maze: m,
            goal_x,
            goal_y,
            rows,
            cols,
        })
    }

    fn process(mut data: Self::Parsed) -> anyhow::Result<()> {
        let res = unsafe {
            pass(
                data.maze.as_mut_ptr(),
                data.goal_x,
                data.goal_y,
                data.rows as i32,
                data.cols as i32,
            )
        };
        println!("result might be something like {res}");

        Ok(())
    }
}

struct Day20 {}

fn main() -> std::process::ExitCode {
    boiler_plate::main_wrap::<Day20>()
}
