#include "cuda_interface.h"
#include "claw.cuh"
#include "cub/util_debug.cuh"
#include <cstdio>
#include "thrust/reduce.h"
#include "thrust/execution_policy.h"
#include "thrust/functional.h"

#include <assert.h>
#define cdpErrchk(ans) \
  { cdpAssert((ans), __FILE__, __LINE__); }
__device__ void cdpAssert(
    cudaError_t code, const char* file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    printf(
        "%s:%d GPU kernel assert %d: %s \n", file, line, code,
        cudaGetErrorString(code));
    if (abort)
      assert(0);
  }
}
#define CubDebugExitPrint(ans) \
  { CubDebugExitPrintImpl((ans), __FILE__, __LINE__); }
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

int pass(int* Ax, int* Bx, int* Ay, int* By, int* Tx, int* Ty, int size) {

  // printf("okay let's do some work for size %d\n", size);

  int* dev_ax;
  int* dev_ay;
  int* dev_bx;
  int* dev_by;
  int* dev_tx;
  int* dev_ty;
  int* res;
  
  CubDebugExit(cudaMalloc(&dev_ax, size * sizeof(int)));
  CubDebugExit(cudaMalloc(&dev_ay, size * sizeof(int)));
  CubDebugExit(cudaMalloc(&dev_bx, size * sizeof(int)));
  CubDebugExit(cudaMalloc(&dev_by, size * sizeof(int)));
  CubDebugExit(cudaMalloc(&dev_tx, size * sizeof(int)));
  CubDebugExit(cudaMalloc(&dev_ty, size * sizeof(int)));
  CubDebugExit(cudaMalloc(&res, size * sizeof(int)));

  // printf("init half done\n");

  CubDebugExit(cudaMemcpy( dev_ax, Ax, size * sizeof(int), cudaMemcpyHostToDevice));
  CubDebugExit(cudaMemcpy( dev_ay, Ay, size * sizeof(int), cudaMemcpyHostToDevice));
  CubDebugExit(cudaMemcpy( dev_bx, Bx, size * sizeof(int), cudaMemcpyHostToDevice));
  CubDebugExit(cudaMemcpy( dev_by, By, size * sizeof(int), cudaMemcpyHostToDevice));
  CubDebugExit(cudaMemcpy( dev_tx, Tx, size * sizeof(int), cudaMemcpyHostToDevice));
  CubDebugExit(cudaMemcpy( dev_ty, Ty, size * sizeof(int), cudaMemcpyHostToDevice));
  
  // printf("init done\n");

  run_claw<<<1, size>>>(dev_ax, dev_bx, dev_ay, dev_by, dev_tx, dev_ty, res);

  // printf("call done\n");

  CubDebugExitPrint(cudaPeekAtLastError());
  CubDebugExitPrint(cudaDeviceSynchronize());
  CubDebugExitPrint(cudaPeekAtLastError());

  // printf("sync done\n");

  auto t = thrust::reduce(thrust::device, res, res+size, 0, thrust::plus<int>());

  CubDebugExitPrint(cudaFree(dev_ax));
  CubDebugExitPrint(cudaFree(dev_ay));
  CubDebugExitPrint(cudaFree(dev_bx));
  CubDebugExitPrint(cudaFree(dev_by));
  CubDebugExitPrint(cudaFree(dev_tx));
  CubDebugExitPrint(cudaFree(dev_ty));
  CubDebugExitPrint(cudaFree(res));


  return t;
}
