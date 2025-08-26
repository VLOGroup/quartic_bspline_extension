#pragma once

#include <cuda_runtime.h>

__constant__ float supp_lower = -2.5f;
__constant__ float supp_rad = 2.5f;
__constant__ float supp_width = 5.0f;   // = 2 * supp_rad
__constant__ int num_supp_intervals = 5;

__constant__ float quartic_bspline_coeffs[5][5] = {
    {  625.0f / 384.0f,  125.0f / 48.0f,  75.0f / 48.0f,  10.0f / 24.0f,  1.0f / 24.0f},
    {    55.0f / 96.0f,   -5.0f / 24.0f, -30.0f / 24.0f, -20.0f / 24.0f, -4.0f / 24.0f},
    {  115.0f / 192.0f,            0.0f, -15.0f / 24.0f,           0.0f,  6.0f / 24.0f},
    {    55.0f / 96.0f,    5.0f / 24.0f, -30.0f / 24.0f,  20.0f / 24.0f, -4.0f / 24.0f},
    {  625.0f / 384.0f, -125.0f / 48.0f,  75.0f / 48.0f, -10.0f / 24.0f,  1.0f / 24.0f}
};