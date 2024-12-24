#include <cstdint>
#include <cstdio>
#include "claw.cuh"

__global__ void run_claw(
    int* Ax, int* Bx, int* Ay, int* By, int* Tx, int* Ty, int* res) {
  res[threadIdx.x] = 0;
  for (std::int32_t tokens = 0; tokens <= 400; tokens++) {
    for (std::int32_t A_pushes = 0; A_pushes <= tokens / 3; A_pushes++) {
      std::int32_t B_pushes = tokens - 3 * A_pushes;
      if (B_pushes > 100)
        continue;
      // printf(
      //     "testing at access at index %d with %d tokens.\nPush A %d, Push B "
      //     "%d\n",
      //     threadIdx.x, tokens, A_pushes, B_pushes);
      auto reaches_x = A_pushes * Ax[threadIdx.x] + B_pushes * Bx[threadIdx.x];
      // printf(
      //     "%d = %d*%d + %d*%d", reaches_x, A_pushes, Ax[threadIdx.x],
      //     B_pushes, Bx[threadIdx.x]);
      auto reaches_y = A_pushes * Ay[threadIdx.x] + B_pushes * By[threadIdx.x];
      // printf(
      //     "this reaches %d, %d (%d, %d)\n", reaches_x, reaches_y,
      //     Tx[threadIdx.x], Ty[threadIdx.x]);

      /// The cheapest way to win the prize is by pushing the A button 80 times
      /// and the B button 40 times. This would line up the claw along the X
      /// axis (because 80*94 + 40*22 = 8400) and along the Y axis (because
      /// 80*34 + 40*67 = 5400). Doing this would cost 80*3 tokens for the A
      /// presses and 40*1 for the B presses, a total of 280 tokens.

      if (Tx[threadIdx.x] != reaches_x)
        continue;
      if (Ty[threadIdx.x] == reaches_y) {
        res[threadIdx.x] = tokens;
        // printf("machine %d can be won with %d tokens\n", threadIdx.x, tokens);
        break;
      }
    }
  }
}
