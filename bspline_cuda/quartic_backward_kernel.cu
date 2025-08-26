#include <vector>

#include <torch/extension.h>

#include "include/constants.cuh"
#include "include/debug_utils.cuh"

template <typename T>
__global__ void quartic_bspline_backward_kernel(
    const torch::PackedTensorAccessor32<T, 2, torch::RestrictPtrTraits> x,
    const torch::PackedTensorAccessor32<T, 2, torch::RestrictPtrTraits> weight_tensor,
    const torch::PackedTensorAccessor32<T, 1, torch::RestrictPtrTraits> centers,
    const T scale,
    const torch::PackedTensorAccessor32<T, 2, torch::RestrictPtrTraits> grad_out,
    torch::PackedTensorAccessor32<T, 2, torch::RestrictPtrTraits> grad_w
){
    const int idx_f = blockIdx.y * blockDim.y + threadIdx.y;
    const int idx_x = blockIdx.x * blockDim.x + threadIdx.x;

    const int num_rows = x.size(0);
    const int num_cols = x.size(1);

    const int num_centers = centers.size(0);

    if (idx_f < num_rows && idx_x < num_cols){
        const T xx = x[idx_f][idx_x];

        for (auto j = 0; j < num_centers; j++){
            const T x_scaled = (xx - centers[j]) / scale;
            if (fabsf(x_scaled) < supp_rad){               
                T x_pwr = 1.0f;
                for (auto i = 0; i < num_supp_intervals; i++){
                    // NOTE
                    //    > atomicAdd() is necessary here, since multiple values of input tensor x
                    //      contribute to the gradient.
                    atomicAdd(&(grad_w[idx_f][i]), grad_out[idx_f][idx_x] * x_pwr);
                    x_pwr *= x_scaled[idx_f][idx_x];
                }
            }
        }
    }
}