#pragma once

#include <algorithm>
#include "constants.h"

#ifdef __CUDACC__
#define HOST_DEVICE_INLINE __host__ __device__ __forceinline__
#else
#define HOST_DEVICE_INLINE inline
#endif

template <typename T>
HOST_DEVICE_INLINE std::pair<int64_t, int64_t> compute_center_index_bounds(
    const T x, 
    const T c_0, 
    const T scale, 
    const T delta_inv, 
    const int num_centers
){
    T center_bound_lower = std::ceil((x - c_0 - SUPP_RAD * scale) * delta_inv);
    const int center_idx_lower = std::clamp(static_cast<int>(center_bound_lower), 
                                               0, num_centers - 1);

    T center_bound_upper = std::floor((x - c_0 + SUPP_RAD * scale) * delta_inv);
    const int center_idx_upper = std::clamp(static_cast<int>(center_bound_upper), 
                                               0, num_centers - 1);

    return {center_idx_lower, center_idx_upper};
}
