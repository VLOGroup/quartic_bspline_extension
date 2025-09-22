#include <vector>
#include <torch/extension.h>

#include "include/constants.h"
#include "include/index_utils.h"
#include "include/device_utils.h"

/**
 * @brief Pure C++ implementation of backward step of 
 *      quartic (midpoint cardinal) b-spline potential
 * 
 * @note
 *  > Fallback if no CUDA device is detected
 *  > OpenMP-based implementation.
 *  > Function takes the same arguments as the corresponding CUDA kernel.
 */
template <typename T>
void quartic_bspline_backward_cpu_kernel(
    const torch::PackedTensorAccessor32<T, 4> x,
    const torch::PackedTensorAccessor32<T, 2> weight_tensor,
    const torch::PackedTensorAccessor32<T, 1> centers,
    const T scale,
    const T scale_inv,
    const T delta_inv,
    const torch::PackedTensorAccessor32<T, 4> grad_out,
    torch::PackedTensorAccessor32<T, 2> grad_w
){
    #pragma omp parallel for collapse(4)
    for (int64_t idx_bs = 0; idx_bs < x.size(0); idx_bs++) {
        for (int64_t idx_f = 0; idx_f < x.size(1); idx_f++) {
            for (int64_t idx_w = 0; idx_w < x.size(2); idx_w++) {
                for (int64_t idx_h = 0; idx_h < x.size(3); idx_h++) {

                    const T x_ = x[idx_bs][idx_f][idx_w][idx_h];
                    const std::pair<int, int> center_idx_bounds =
                        compute_center_index_bounds(x_, centers[0], scale, delta_inv, centers.size(0));

                    for (int j = center_idx_bounds.first; j <= center_idx_bounds.second; j++) {
                        const T x_scaled = (x_ - centers[j]) * scale_inv;
                        if (fabs(x_scaled) < SUPP_RAD) {
                            int interval = static_cast<int>(x_scaled - SUPP_LOWER);
                            interval = std::max(0, std::min(NUM_SUPP_INTERVALS - 1, interval));

                            T spline_val = QUARTIC_BSPLINE_COEFFS[interval][4];
                            for (int k = 1; k <= NUM_SUPP_INTERVALS - 1; k++) {
                                spline_val = spline_val * x_scaled +
                                            QUARTIC_BSPLINE_COEFFS[interval][NUM_SUPP_INTERVALS - 1 - k];
                            }

                            // atomic operation to update grad_w
                            #pragma omp atomic
                            grad_w[idx_f][j] += grad_out[idx_bs][idx_f][idx_w][idx_h] * spline_val;
                        }
                    }
                }
            }
        }
    }
}


std::vector<torch::Tensor> quartic_bspline_backward_cpu_function(
    const torch::Tensor x,
    const torch::Tensor weight_tensor,
    const torch::Tensor centers,
    const double scale,
    const torch::Tensor grad_out
){
    check_device_and_datatype({x, weight_tensor, centers, grad_out});

    auto scalar_type = x.scalar_type();
    auto grad_w = torch::zeros_like(weight_tensor);

    const double scale_inv = 1.0 / scale;
    const double delta_inv = 1.0 / (centers[1].item<double>() - centers[0].item<double>());

    AT_DISPATCH_FLOATING_TYPES(scalar_type, "quartic_bspline_backward_cpu", [&] {
        quartic_bspline_backward_cpu_kernel<scalar_t>(
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

    return {grad_w};
}