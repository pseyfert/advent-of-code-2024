#include "claw.cuh"

__global__ void run_claw(
    int* Ax, int* Bx, int* Ay, int* By, int* Tx, int* Ty, int* res) {
  res[threadIdx.x] = Ax[threadIdx.x] + Bx[threadIdx.x];
}
