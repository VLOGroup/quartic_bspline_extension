#include <vector>
#include <torch/extension.h>
#include <omp.h>

#include "include/constants.h"
#include "include/index_utils.h"
#include "include/device_utils.h"

/**
 * @brief C++ kernel performing the forward step of the spline potential for
 *      quartic (midpoint cardinal) b-spline.
 * 
 * @note
 *  > Acts as fallback for quartic_bspline_forward_cuda_kernel() if no CUDA device
 *      is detected.
 *  > OpenMP-based implementation.
 *  > Takes the same arguments as the corresponding CUDA kernel.
 */
template <typename T>
void quartic_bspline_forward_cpu_kernel(
    const torch::PackedTensorAccessor32<T, 4> x,
    const torch::PackedTensorAccessor32<T, 2> weight_tensor,
    const torch::PackedTensorAccessor32<T, 1> centers,
    const T scale,
    const T scale_inv,
    const T delta_inv,
    torch::PackedTensorAccessor32<T, 4> rho,
    torch::PackedTensorAccessor32<T, 4> rho_prime
){
    #pragma omp parallel for collapse(4)
    for (int64_t idx_bs = 0; idx_bs < x.size(0); idx_bs++) {
        for (int64_t idx_f = 0; idx_f < x.size(1); idx_f++) {
            for (int64_t idx_w = 0; idx_w < x.size(2); idx_w++) {
                for (int64_t idx_h = 0; idx_h < x.size(3); idx_h++) {
                    T rho_val = 0.0f;
                    T rho_prime_val = 0.0f;
                    
                    const T x_ = x[idx_bs][idx_f][idx_w][idx_h];
                    const std::pair<int, int> center_idx_bounds =
                        compute_center_index_bounds(x_, centers[0], scale, delta_inv, centers.size(0));

                    for (int j = center_idx_bounds.first; j <= center_idx_bounds.second; j++){
                        const T x_scaled = (x_ - centers[j]) * scale_inv;

                        if (fabsf(x_scaled) < SUPP_RAD){
                            int interval = (int)(x_scaled - SUPP_LOWER);
                            interval = std::max(0, std::min(NUM_SUPP_INTERVALS - 1, interval));

                            // evaluate local spline and its derivative
                            T spline_val = QUARTIC_BSPLINE_COEFFS[interval][4];
                            T spline_deriv = 0.0f;
                            for (int k = 1; k <= NUM_SUPP_INTERVALS - 1; k++){
                                spline_deriv = spline_deriv * x_scaled + spline_val;
                                spline_val = spline_val * x_scaled 
                                        + QUARTIC_BSPLINE_COEFFS[interval][NUM_SUPP_INTERVALS - 1 - k];
                            }
                            rho_val += weight_tensor[idx_f][j] * spline_val;
                            rho_prime_val += weight_tensor[idx_f][j] * spline_deriv * scale_inv;                
                        }
                    }
                    rho[idx_bs][idx_f][idx_w][idx_h] = rho_val;
                    rho_prime[idx_bs][idx_f][idx_w][idx_h] = rho_prime_val;
                }
            }
        }
    }
}


std::vector<torch::Tensor> quartic_bspline_forward_cpu_function(
    const torch::Tensor x,
    const torch::Tensor weight_tensor,
    const torch::Tensor centers,
    const double scale
){
    check_device_and_datatype({x, weight_tensor, centers});

    auto scalar_type = x.scalar_type();

    auto rho = torch::empty_like(x);
    auto rho_prime = torch::empty_like(x);

    const double scale_inv = 1.0 / scale;
    const double delta_inv = 1.0 / (centers[1].item<double>() - centers[0].item<double>());

    AT_DISPATCH_FLOATING_TYPES(scalar_type, "quartic_bspline_forward_cpu", [&] {
        quartic_bspline_forward_cpu_kernel<scalar_t>(
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

    return {rho, rho_prime};
}