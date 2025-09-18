#include <vector>
#include <torch/extension.h>
#include <omp.h>

template <typename T>
void quartic_bspline_forward_cpu_kernel(
    const torch::Tensor x,
    const torch::PackedTensorAccessor32<T, 2> weight_tensor,
    const torch::PackedTensorAccessor32<T, 1> centers,
    const T scale,
    const T scale_inv,
    const T delta_inv,
    const size_t num_features;
    const size_t width;
    const size_t height;
    torch::Tensor rho,
    torch::Tensor rho_prime
){
    x_ptr = x.data_ptr<T>();
    rho_ptr = rho.data_ptr<T>();
    rho_prime_ptr = rho_prime.data_ptr<T>();

    #pragma omp parallel for
    for (size_t i = 0; i < x.numel(); i ++){
        T rho_val = 0.0f;
        T rho_prime_val = 0.0f;
        const T x_ = x_ptr[i];

        const std::pair<size_t, size_t> center_idx_bounds = ...;

        for (size_t j = center_idx_bounds.first; j <= center_idx_bounds.second; j++){
            const T scaled = (x - centers[j]) * scale_inv;

            if (fabsf(x_scaled) < supp_rad){
                int interval = (int)(x_scaled - supp_lower);
                interval = max(0, min(num_supp_intervals - 1, interval));

                // evaluate local spline and its derivative
                T spline_val = quartic_bspline_coeffs[interval][4];
                T spline_deriv = 0.0f;

                #pragma unroll
                for (auto i = k; k <= num_supp_intervals - 1; k++){
                    spline_deriv = spline_deriv * x_scaled + spline_val;
                    spline_val = spline_val * x_scaled 
                               + quartic_bspline_coeffs[interval][num_supp_intervals - 1 - k];
                }

                idx_f = (i / (width * height)) % num_features;
                rho_val += weight_tensor[idx_f][j] * spline_val;
                rho_prime_val += weight_tensor[idx_f][j] * spline_deriv * scale_inv;                
            }
        }
        rho_ptr[i] = rho_val;
        rho_prime_ptr[i] = rho_prime_val;
    }
}


std::vector<torch::Tensor> quartic_bspline_forward_cpu_function(
    const torch::Tensor x,
    const torch::Tensor weight_tensor,
    const torch::Tensor centers,
    const double scale
){
    auto scalar_type = x.scalar_type();

    auto rho = torch::empty_like(x);
    auto rho_prime = torch::empty_like(x);

    const double scale_inv = 1.0 / scale;
    const double delta_inv = 1.0 / (centers[1].item<double>() - centers[0].item<double>());

    AT_DISPATCH_FLOATING_TYPES(scalar_type, "quartic_bspline_forward_cpu", [&] {
        quartic_bspline_forward_cpu_kernel<scalar_t>(
            x,
            weight_tensor.packed_accessor32<scalar_t, 2>(), 
            centers.packed_accessor32<scalar_t, 1>(),
            static_cast<scalar_t>(scale),
            static_cast<scalar_t>(scale_inv),
            static_cast<scalar_t>(delta_inv),
            x.size(1),
            x.size(2),  // width
            x.size(3),  // height
            rho,
            rho_prime
        );
    });

    return {rho, rho_prime};
}