#include <vector>
#include <torch/extension.h>

#include "include/constants.cuh"
#include "include/debug_utils.cuh"
#include "include/index_utils.cuh"
#include "include/device_utils.cuh"

/**
 * @brief CUDA kernel which performs evaluation of quartic (midpoint cardinal) b-spline potential function 
 *      and its derivative.
 * 
 * @note 
 *  > Assumption 1: Along every feature dimension, the same set of center nodes are used
 *  > Assumption 2: Center nodes are equally spaced in an interval [a, b]
 * 
 * @tparam T Floating point type: float, or double.
 * @param x Tensor of shape [bs, f, w, h] at which the b-spline has to be evaluated.
 * @param weight_tensor Tensor of shape [f, num_centers] corresponding to the weights spline potential
 *      at the center nodes for each marginal.
 * @param centers Tensor of shape [num_centers, ] of center nodes.
 * @param scale Scaling parameter.
 * @param scale_inv Inverse of the scaling parameter.
 * @param delta_inv Inverse of distance between (equally spaced) center nodes.
 * @param rho Tensor of shape [bs, f, w, h] - the evaluation of the spline potenital at x.
 * @param rho_prime Tensor of shape [bs, f, w, h] - the state-derivative of the spline potential 
 *      evaluated at x.
 */
template <typename T>
__global__ void quartic_bspline_forward_kernel(
    const torch::PackedTensorAccessor32<T, 4> x,
    const torch::PackedTensorAccessor32<T, 2> weight_tensor,
    const torch::PackedTensorAccessor32<T, 1> centers,
    const T scale,
    const T scale_inv,
    const T delta_inv,
    torch::PackedTensorAccessor32<T, 4> rho,
    torch::PackedTensorAccessor32<T, 4> rho_prime
){
    const int idx_h = blockIdx.x * blockDim.x + threadIdx.x;
    const int idx_w = blockIdx.y * blockDim.y + threadIdx.y;
    const int idx_bf = blockIdx.z;

    const size_t num_features = x.size(1);
    const int idx_bs = idx_bf / num_features;
    const int idx_f = idx_bf % num_features;

    if (idx_bs < x.size(0) && idx_f < num_features && idx_w < x.size(2) && idx_h < x.size(3)){
        T rho_val = 0.0f;
        T rho_prime_val = 0.0f;
        const T x_ = x[idx_bs][idx_f][idx_w][idx_h];

        /**
         * Compute center index range corresponding to the current value of x - note that the
         * computation relies on the assumption of equally spaced center nodes!
         */
        const std::pair<size_t, size_t> center_idx_bounds = 
                    compute_center_index_bounds(x_, centers[0], scale, delta_inv, centers.size(0));

        for (size_t j = center_idx_bounds.first; j <= center_idx_bounds.second; j++){
            const T x_scaled = (x_ - centers[j]) * scale_inv;

            if (fabsf(x_scaled) < supp_rad){
                // determine support interval
                int interval = (int)(x_scaled - supp_lower);
                interval = max(0, min(num_supp_intervals - 1, interval));

                // evaluate local spline and its derivative
                T spline_val = quartic_bspline_coeffs[interval][4];
                T spline_deriv = 0.0f;

                #pragma unroll
                for (auto i = 1; i <= num_supp_intervals - 1; i++){
                    spline_deriv = spline_deriv * x_scaled + spline_val;
                    spline_val = spline_val * x_scaled 
                               + quartic_bspline_coeffs[interval][num_supp_intervals - 1 - i];
                }

                rho_val += weight_tensor[idx_f][j] * spline_val;
                rho_prime_val += weight_tensor[idx_f][j] * spline_deriv * scale_inv;
            }
        }
        rho[idx_bs][idx_f][idx_w][idx_h] = rho_val;
        rho_prime[idx_bs][idx_f][idx_w][idx_h] = rho_prime_val;
    }
}

std::vector<torch::Tensor> quartic_bspline_forward_function(
    const torch::Tensor x,
    const torch::Tensor weight_tensor,
    const torch::Tensor centers,
    const double scale
){
    check_device_and_datatype({x, weight_tensor, centers});

    const dim3 block_size(32, 8);
    const dim3 grid_size((x.size(3) + block_size.x - 1) / block_size.x, 
                         (x.size(2) + block_size.y - 1) / block_size.y,
                         x.size(0) * x.size(1));

    auto scalar_type = x.scalar_type();

    auto rho = torch::empty_like(x);
    auto rho_prime = torch::empty_like(x);

    const double scale_inv = 1.0 / scale;
    const double delta_inv = 1.0 / (centers[1].item<double>() - centers[0].item<double>());

    AT_DISPATCH_FLOATING_TYPES(scalar_type, "quartic_bspline_forward", [&] {
        quartic_bspline_forward_kernel<scalar_t><<<grid_size, block_size>>>(
            x.packed_accessor32<scalar_t, 4>(),
            weight_tensor.packed_accessor32<scalar_t, 2>(), 
            centers.packed_accessor32<scalar_t, 1>(),
            static_cast<scalar_t>(scale),
            static_cast<scalar_t>(scale_inv),
            static_cast<scalar_t>(delta_inv),
            rho.packed_accessor32<scalar_t, 4>(),
            rho_prime.packed_accessor32<scalar_t, 4>()
        );
    });

    CUDA_DEBUG_FUNC(cudaGetLastError());

    return {rho, rho_prime};
}