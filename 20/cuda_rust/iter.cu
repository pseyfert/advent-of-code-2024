#include "iter.cuh"
#define CUB_STDERR
#include <cooperative_groups.h>
#include <cub/util_debug.cuh>
#include <cuda/std/barrier>
#include <experimental/mdspan>

__device__ cuda::barrier<cuda::thread_scope_block> barrier;

using myspan = std::experimental::mdspan<
    int, std::experimental::extents<
             size_t, std::experimental::dynamic_extent,
             std::experimental::dynamic_extent>>;

__device__ bool good(int x) {
  return x >= 0 && x < cuda::std::numeric_limits<int>::max();
}

__device__ bool good_cheat(int x, int cur) {
  return x >= 0 && x < cuda::std::numeric_limits<int>::max() && x < cur - 2;
}

__global__ void doit(
    int* grid, int* down_cheat, int* up_cheat, int* left_cheat,
    int* right_cheat, int* goal_x, int* goal_y) {
  auto block = cooperative_groups::this_thread_block();
  printf("%d,%d", blockDim.x , blockDim.y);
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    init(&barrier, blockDim.x * blockDim.y);
  }

  myspan maze(grid, blockDim.y, blockDim.x);
  myspan c_r(right_cheat, blockDim.y, blockDim.x);
  myspan c_u(up_cheat, blockDim.y, blockDim.x);
  myspan c_d(down_cheat, blockDim.y, blockDim.x);
  myspan c_l(left_cheat, blockDim.y, blockDim.x);

  // cuda::std::numeric_limits<int>::max() = unreached;
  // -1                                    = wall;
  //  0                                    = start;

  block.sync();
  CubDebug(cudaPeekAtLastError());

  barrier.arrive_and_wait();

  for (std::int64_t i = 1;; ++i) {
    if ((maze)(*goal_y, *goal_x) < cuda::std::numeric_limits<int>::max()) {
      barrier.arrive_and_drop();
      break;
    } else if (
        (maze)(threadIdx.y, threadIdx.x) <
        cuda::std::numeric_limits<int>::max()) {
      // This drops all visited and wall spaces
      barrier.arrive_and_drop();
      break;
    } else {
      int reach = cuda::std::numeric_limits<int>::max();
      if (threadIdx.x > 0) {
        if (auto lookup = (maze)(threadIdx.y, threadIdx.x - 1); good(lookup)) {
          reach = lookup + 1;
        }
      }
      if (threadIdx.y > 0) {
        if (auto lookup = (maze)(threadIdx.y - 1, threadIdx.x); good(lookup)) {
          reach = lookup + 1;
        }
      }
      if (threadIdx.x < gridDim.x - 1) {
        if (auto lookup = (maze)(threadIdx.y, threadIdx.x + 1); good(lookup)) {
          reach = lookup + 1;
        }
      }
      if (threadIdx.y < gridDim.y - 1) {
        if (auto lookup = (maze)(threadIdx.y + 1, threadIdx.x); good(lookup)) {
          reach = lookup + 1;
        }
      }
      (maze)(threadIdx.y, threadIdx.x) = reach;
      barrier.arrive_and_wait();
      if (reach < cuda::std::numeric_limits<int>::max()) {
        if (threadIdx.x > 1) {
          if (auto lookup = (maze)(threadIdx.y, threadIdx.x - 2);
              good_cheat(lookup, reach)) {
            (c_r)(threadIdx.y, threadIdx.x) = lookup - reach - 2;
          }
        }
        if (threadIdx.y > 1) {
          if (auto lookup = (maze)(threadIdx.y - 1, threadIdx.x);
              good_cheat(lookup, reach)) {
            (c_d)(threadIdx.y, threadIdx.x) = lookup - reach - 2;
          }
        }
        if (threadIdx.x < gridDim.x - 2) {
          if (auto lookup = (maze)(threadIdx.y, threadIdx.x + 2);
              good_cheat(lookup, reach)) {
            (c_l)(threadIdx.y, threadIdx.x) = lookup - reach - 2;
          }
        }
        if (threadIdx.y < gridDim.y - 2) {
          if (auto lookup = (maze)(threadIdx.y + 2, threadIdx.x);
              good_cheat(lookup, reach)) {
            (c_u)(threadIdx.y, threadIdx.x) = lookup - reach - 2;
          }
        }
      }
      barrier.arrive_and_wait();
    }
  }
}
