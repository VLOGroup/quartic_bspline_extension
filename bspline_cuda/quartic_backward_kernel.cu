#include <vector>

#include <torch/extension.h>

#include "include/constants.cuh"
#include "include/debug_utils.cuh"

template <typename T>
__global__ void quartic_bspline_backward_kernel(
    const torch::PackedTensorAccessor32<T, 2> x,
    const torch::PackedTensorAccessor32<T, 2> weight_tensor,
    const torch::PackedTensorAccessor32<T, 1> centers,
    const T scale,
    const torch::PackedTensorAccessor32<T, 2> grad_out,
    torch::PackedTensorAccessor32<T, 2> grad_w
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
                
                // determine support interval
                int interval = (int)(x_scaled - supp_lower);
                interval = max(0, min(num_supp_intervals - 1, interval));
                
                // evaluate local spline
                T spline_val = quartic_bspline_coeffs[interval][4];

                #pragma unroll
                for (auto i = 1; i <= num_supp_intervals - 1; i++){
                    spline_val = spline_val * x_scaled + quartic_bspline_coeffs[interval][num_supp_intervals - 1 - i];
                }

                atomicAdd(&(grad_w[idx_f][j]), grad_out[idx_f][idx_x] * spline_val);
            }
        }
    }
}

std::vector<torch::Tensor> quartic_bspline_backward_function(
    const torch::Tensor x,
    const torch::Tensor weight_tensor,
    const torch::Tensor centers,
    const float scale,
    const torch::Tensor grad_out
){
    unsigned int bs = x.size(0);
    unsigned int f = x.size(1);
    unsigned int w = x.size(2);
    unsigned int h = x.size(3);
    auto x_ = torch::permute(x, {1, 0, 2, 3}).contiguous().reshape({f, bs * w * h});
    auto grad_out_ = torch::permute(grad_out, {1, 0, 2, 3}).contiguous().reshape({f, bs * w * h});

    const dim3 block_size(1024, 1, 1);
    const dim3 num_blocks((x_.size(1) + block_size.x - 1) / block_size.x, 
                          (x_.size(0) + block_size.y - 1) / block_size.y);

    auto scalar_type = x.scalar_type();
    /*
        NOTE
        ----
            > Initialisation with zero is important here!
     */
    auto grad_w = torch::zeros_like(weight_tensor);

    AT_DISPATCH_FLOATING_TYPES(scalar_type, "quartic_bspline_backward", [&] {
        quartic_bspline_backward_kernel<scalar_t><<<num_blocks, block_size>>>(
            x_.packed_accessor32<scalar_t, 2>(),
            weight_tensor.packed_accessor32<scalar_t, 2>(), 
            centers.packed_accessor32<scalar_t, 1>(),
            static_cast<scalar_t>(scale),
            grad_out_.packed_accessor32<scalar_t, 2>(),
            grad_w.packed_accessor32<scalar_t, 2>()
        );
    });

    CUDA_DEBUG_FUNC(cudaGetLastError());

    return {grad_w.contiguous()};
}