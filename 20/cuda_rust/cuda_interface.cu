#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include "cuda_interface.h"
#include "iter.cuh"

#define CubDebugExitPrint(ans)                        \
  {                                                   \
    CubDebugExitPrintImpl((ans), __FILE__, __LINE__); \
  }
__host__ void CubDebugExitPrintImpl(
    cudaError_t code, const char* file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    printf(
        "%s:%d GPU kernel assert %d: %s \n", file, line, code,
        cudaGetErrorString(code));
    if (abort)
      CubDebugExit(code);
  }
}

int run(int* grid, int goal_x, int goal_y, int rows, int cols) {
  thrust::device_vector<int> left_cheat(rows * cols, 0);
  thrust::device_vector<int> right_cheat(rows * cols, 0);
  thrust::device_vector<int> up_cheat(rows * cols, 0);
  thrust::device_vector<int> down_cheat(rows * cols, 0);
  thrust::device_vector<int> dev_grid(rows * cols);

  int* dev_goal_x;
  int* dev_goal_y;
  CubDebugExit(cudaMallocManaged(&dev_goal_x, sizeof(int)));
  CubDebugExit(cudaMallocManaged(&dev_goal_y, sizeof(int)));
  *dev_goal_x = 0;
  *dev_goal_y = 1;

  CubDebugExit(cudaMemcpy(
      thrust::raw_pointer_cast(dev_grid.data()), grid,
      rows * cols * sizeof(int), cudaMemcpyHostToDevice));

  printf("Trying to launch %d threads\n", cols*rows);
  // {blockDim.x, blockDim.y}
  doit<<<1, {cols, rows}>>>(
      thrust::raw_pointer_cast(dev_grid.data()),
      thrust::raw_pointer_cast(down_cheat.data()),
      thrust::raw_pointer_cast(up_cheat.data()),
      thrust::raw_pointer_cast(left_cheat.data()),
      thrust::raw_pointer_cast(right_cheat.data()), dev_goal_x, dev_goal_y);

  CubDebugExitPrint(cudaPeekAtLastError());
  CubDebugExitPrint(cudaDeviceSynchronize());
  CubDebugExitPrint(cudaPeekAtLastError());

  thrust::device_vector<int> best_cheat(4);

  best_cheat[0] = *thrust::max_element(
      thrust::device, left_cheat.begin(), left_cheat.end());
  best_cheat[1] = *thrust::max_element(
      thrust::device, right_cheat.begin(), right_cheat.end());
  best_cheat[2] =
      *thrust::max_element(thrust::device, up_cheat.begin(), up_cheat.end());
  best_cheat[3] = *thrust::max_element(
      thrust::device, down_cheat.begin(), down_cheat.end());
  auto really_best_cheat = *thrust::max_element(
      thrust::device, best_cheat.begin(), best_cheat.end());
  return thrust::count(
             thrust::device, left_cheat.begin(), left_cheat.end(),
             really_best_cheat) +
         thrust::count(
             thrust::device, down_cheat.begin(), down_cheat.end(),
             really_best_cheat) +
         thrust::count(
             thrust::device, up_cheat.begin(), up_cheat.end(),
             really_best_cheat) +
         thrust::count(
             thrust::device, right_cheat.begin(), right_cheat.end(),
             really_best_cheat);
}
