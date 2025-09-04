#pragma once

#include "constants.cuh"

template <typename T>
__device__ __forceinline__ std::pair<size_t, size_t> compute_center_index_bounds(
    const T x, 
    const T c_0, 
    const T scale, 
    const T delta_inv, 
    const size_t num_centers
){
    T center_bound_lower = std::ceil((x - c_0 - supp_rad * scale) * delta_inv);
    center_bound_lower = center_bound_lower < 0 ? 0 : center_bound_lower;
    const size_t center_idx_lower = static_cast<size_t>(center_bound_lower);

    T center_bound_upper = std::floor((x - c_0 + supp_rad * scale) * delta_inv);
    center_bound_upper = center_bound_upper < num_centers ? center_bound_upper : num_centers -1;
    const size_t center_idx_upper = static_cast<size_t>(center_bound_upper);

    return {center_idx_lower, center_idx_upper};
}
