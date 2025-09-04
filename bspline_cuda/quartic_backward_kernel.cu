#include <vector>
#include <torch/extension.h>

#include "include/constants.cuh"
#include "include/debug_utils.cuh"
#include "include/index_utils.cuh"
#include "include/device_utils.cuh"

/**
 * @brief CUDA kernel implementing the backward step of quartic (midpoint cardinal) 
 *      b-spline potential functions. It computes the derivative of the potential
 *      w.r.t. the weights.
 *
 * @note 
 *  > The derivative w.r.t. the state is computed within the kernel for the 
 *    forward step. Aggregation is managed within custom autograd function on 
 *    PyTorch side.
 *  > Optimisation in terms of speed:
 *      * Introduce shared memory on CUDA thread block
 *      * Compute partial gradients on thread block and store results into 
 *        shared memory first.
 *      * Write to global memory only after all threads of the block finished
 *        their work.  
 * 
 * @tparam T Floating point type: float, or double.
 * @param x Tensor of shape [bs, f, w, h] at which the b-spline has to be evaluated.
 * @param weight_tensor Tensor of shape [f, num_centers] corresponding to the 
 *      weights spline potential at the center nodes for each marginal.
 * @param centers Tensor of shape [num_centers, ] of center nodes.
 * @param scale Scaling parameter.
 * @param scale_inv Inverse of the scaling parameter.
 * @param delta_inv Inverse of distance between (equally spaced) center nodes.
 * @param grad_out Tensor of shape [bs, f, w, h] corresponding to the gradient 
 *      of (scalar) loss w.r.t. the output of spline potential.
 * @param grad_w Tensor of shape [f, num_centers] of derivatives w.r.t. to 
 *      the weights of the spline potential evaluated at weight_tensor.
 */
template <typename T>
__global__ void quartic_bspline_backward_kernel(
    const torch::PackedTensorAccessor32<T, 4> x,
    const torch::PackedTensorAccessor32<T, 2> weight_tensor,
    const torch::PackedTensorAccessor32<T, 1> centers,
    const T scale,
    const T scale_inv,
    const T delta_inv,
    const torch::PackedTensorAccessor32<T, 4> grad_out,
    torch::PackedTensorAccessor32<T, 2> grad_w
){
    const int num_centers = centers.size(0);
    const int num_features = x.size(1);

    const int idx_h = blockIdx.x * blockDim.x + threadIdx.x;
    const int idx_w = blockIdx.y * blockDim.y + threadIdx.y;
    const int idx_bf = blockIdx.z;

    const int idx_bs = idx_bf / num_features;
    const int idx_f = idx_bf % num_features;

    if (idx_bs < x.size(0) && idx_f < num_features && idx_w < x.size(2) && idx_h < x.size(3)){

        const T x_ = x[idx_bs][idx_f][idx_w][idx_h];

        const std::pair<size_t, size_t> center_idx_bounds = 
                    compute_center_index_bounds(x_, centers[0], scale, delta_inv, centers.size(0));

        for (auto j = center_idx_bounds.first; j <= center_idx_bounds.second; j++){
            const T x_scaled = (x_ - centers[j]) * scale_inv;
            if (fabsf(x_scaled) < supp_rad){               
                
                // determine support interval
                int interval = (int)(x_scaled - supp_lower);
                interval = max(0, min(num_supp_intervals - 1, interval));
                
                // evaluate local spline
                T spline_val = quartic_bspline_coeffs[interval][4];

                #pragma unroll
                for (auto i = 1; i <= num_supp_intervals - 1; i++){
                    spline_val = spline_val * x_scaled 
                               + quartic_bspline_coeffs[interval][num_supp_intervals - 1 - i];
                }

                atomicAdd(&grad_w[idx_f][j], grad_out[idx_bs][idx_f][idx_w][idx_h] * spline_val);
            }
        }
    }
}

std::vector<torch::Tensor> quartic_bspline_backward_function(
    const torch::Tensor x,
    const torch::Tensor weight_tensor,
    const torch::Tensor centers,
    const double scale,
    const torch::Tensor grad_out
){
    check_device_and_datatype({x, weight_tensor, centers, grad_out});

    const dim3 block_size(32, 8);
    const dim3 grid_size((x.size(3) + block_size.x - 1) / block_size.x, 
                         (x.size(2) + block_size.y - 1) / block_size.y,
                         x.size(0) * x.size(1));

    // Initialisation with zero is important here!!
    auto grad_w = torch::zeros_like(weight_tensor);

    const double scale_inv = 1.0 / scale;
    const double delta_inv = 1.0 / (centers[1].item<double>() - centers[0].item<double>());

    auto scalar_type = x.scalar_type();
    const size_t shared_mem_size = grad_w.numel() * sizeof(scalar_type);
    AT_DISPATCH_FLOATING_TYPES(scalar_type, "quartic_bspline_backward", [&] {
        quartic_bspline_backward_kernel<scalar_t><<<grid_size, block_size, shared_mem_size>>>(
            x.packed_accessor32<scalar_t, 4>(),
            weight_tensor.packed_accessor32<scalar_t, 2>(), 
            centers.packed_accessor32<scalar_t, 1>(),
            static_cast<scalar_t>(scale),
            static_cast<scalar_t>(scale_inv),
            static_cast<scalar_t>(delta_inv),
            grad_out.packed_accessor32<scalar_t, 4>(),
            grad_w.packed_accessor32<scalar_t, 2>()
        );
    });

    CUDA_DEBUG_FUNC(cudaGetLastError());

    return {grad_w};
}