extern "C" {
#include "c_interface.h"
}
#include "cuda_interface.h"
int pass(const int* grid, int goal_x, int goal_y, int rows, int cols) {
  return run(grid, goal_x, goal_y, rows, cols);
}
