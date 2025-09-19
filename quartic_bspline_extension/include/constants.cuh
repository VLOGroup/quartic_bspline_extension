#pragma once

#ifdef __CUDACC__
    #include <cuda_runtime.h>

    __constant__ float SUPP_LOWER = -2.5f;
    __constant__ float SUPP_RAD = 2.5f;
    __constant__ float SUPP_WIDTH = 5.0f;   // = 2 * supp_rad
    __constant__ int NUM_SUPP_INTERVALS = 5;

    __constant__ float QUARTIC_BSPLINE_COEFFS[5][5] = {
        {  625.0f / 384.0f,  125.0f / 48.0f,  75.0f / 48.0f,  10.0f / 24.0f,  1.0f / 24.0f},
        {    55.0f / 96.0f,   -5.0f / 24.0f, -30.0f / 24.0f, -20.0f / 24.0f, -4.0f / 24.0f},
        {  115.0f / 192.0f,            0.0f, -15.0f / 24.0f,           0.0f,  6.0f / 24.0f},
        {    55.0f / 96.0f,    5.0f / 24.0f, -30.0f / 24.0f,  20.0f / 24.0f, -4.0f / 24.0f},
        {  625.0f / 384.0f, -125.0f / 48.0f,  75.0f / 48.0f, -10.0f / 24.0f,  1.0f / 24.0f}
    };
#else
    constexpr float SUPP_LOWER = -2.5f;
    constexpr float SUPP_RAD = 2.5f;
    constexpr float SUPP_WIDTH = 5.0f;   // = 2 * supp_rad
    constexpr int NUM_SUPP_INTERVALS = 5;

    constexpr float QUARTIC_BSPLINE_COEFFS[5][5] = {
        {  625.0f / 384.0f,  125.0f / 48.0f,  75.0f / 48.0f,  10.0f / 24.0f,  1.0f / 24.0f},
        {    55.0f / 96.0f,   -5.0f / 24.0f, -30.0f / 24.0f, -20.0f / 24.0f, -4.0f / 24.0f},
        {  115.0f / 192.0f,            0.0f, -15.0f / 24.0f,           0.0f,  6.0f / 24.0f},
        {    55.0f / 96.0f,    5.0f / 24.0f, -30.0f / 24.0f,  20.0f / 24.0f, -4.0f / 24.0f},
        {  625.0f / 384.0f, -125.0f / 48.0f,  75.0f / 48.0f, -10.0f / 24.0f,  1.0f / 24.0f}
    };
