#pragma once
__global__ void doit(
    int* grid, int* down_cheat, int* up_cheat, int* left_cheat,
    int* right_cheat, int* goal_x, int* goal_y);
