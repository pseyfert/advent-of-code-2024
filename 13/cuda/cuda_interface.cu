#include "cuda_interface.h"
#include "claw.cuh"

int pass(int* Ax, int* Bx, int* Ay, int* By, int* Tx, int* Ty, int size) {
  int *res = (int*)malloc(size * sizeof(int));
  run_claw<<<size, 1>>>(Ax, Bx, Ay, By, Tx, Ty, res);
  int rv = res[0];
  free(res);
  return rv;
}
