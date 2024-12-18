extern "C" {
#include "c_interface.h"
}
#include "cuda_interface.h"

int wrap(int* Ax, int* Bx, int* Ay, int* By, int* Tx, int* Ty, int size) {
  return pass(Ax, Bx, Ay, By, Tx, Ty, size);
}
