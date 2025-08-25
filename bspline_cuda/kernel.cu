#include <iostream>
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void hello_kernel(float *out, int N){
    // Very simple CUDA kernel
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N){
        out[idx] = idx;
        printf("Hello there from CUDA and thread %d!\n", idx);
    }
}

__constant__ float supp_lower = -2.5f;
__constant__ float supp_width = 5.0f;
__constant__ int num_supp_intervals = 5;

__constant__ float quartic_bspline_coeffs[5][5] = {
    {  625.0f / 384.0f,  125.0f / 48.0f,  75.0f / 48.0f,  10.0f / 24.0f,  1.0f / 24.0f},
    {    55.0f / 96.0f,   -5.0f / 24.0f, -30.0f / 24.0f, -20.0f / 24.0f, -4.0f / 24.0f},
    {  115.0f / 192.0f,            0.0f, -15.0f / 24.0f,           0.0f,  6.0f / 24.0f},
    {    55.0f / 96.0f,    5.0f / 24.0f, -30.0f / 24.0f,  20.0f / 24.0f, -4.0f / 24.0f},
    {  625.0f / 384.0f, -125.0f / 48.0f,  75.0f / 48.0f, -10.0f / 24.0f,  1.0f / 24.0f}
};

template <typename T>
__global__ void quartic_bspline_forward_kernel(
    const torch::PackedTensorAccessor32<T, 2, torch::RestrictPtrTraits> x,
    const torch::PackedTensorAccessor32<T, 2, torch::RestrictPtrTraits> weight_tensor,
    const torch::PackedTensorAccessor32<T, 1, torch::RestrictPtrTraits> centers,
    const T scale,
    torch::PackedTensorAccessor32<T, 2, torch::RestrictPtrTraits> rho,
    torch::PackedTensorAccessor32<T, 2, torch::RestrictPtrTraits> rho_prime
){
    const int idx_f = blockIdx.x * blockDim.x + threadIdx.x;
    const int idx_x = blockIdx.y * blockDim.y + threadIdx.y;

    const int num_rows = x.size(0);
    const int num_cols = x.size(1);

    const int num_centers = centers.size(0);

    const T xx = x[idx_f][idx_x];

    T rho_val = 0.0f;
    T rho_prime_val = 0.0f;

    for (auto j = 0; j < num_centers; j++){
        const T x_scaled = (xx - centers[j]) / scale;

        if (fabsf(x_scaled) < 2.5f){

            // determine support interval
            int interval = (int)((x_scaled - supp_lower) / supp_width);
            interval = max(0, min(num_supp_intervals - 1, interval));
            
            // evaluate local spline and its derivative
            T spline_val = quartic_bspline_coeffs[interval][4];
            T spline_deriv = 0.0f;
            
            #pragma unroll
            for (auto i = 1; i < num_supp_intervals - 1; i++){
                spline_deriv = spline_deriv * x_scaled + spline_val;
                spline_val = spline_val * x_scaled + quartic_bspline_coeffs[interval][num_supp_intervals - 1 - i];
            }

            rho_val += weight_tensor[idx_f][j] * spline_val;
            rho_prime_val += weight_tensor[idx_f][j] * spline_deriv;
        }
    }

    rho[idx_f][idx_x] = rho_val;
    rho_prime[idx_f][idx_x] = rho_prime_val;
}

std::vector<torch::Tensor> quartic_bspline_forward_function(
    torch::Tensor x,
    torch::Tensor weight_tensor,
    torch::Tensor centers,
    float scale
){
    const int N = x.numel();
    const int num_threads_per_block = 256;
    const int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    auto scalar_type = x.scalar_type();

    unsigned int bs = x.size(0);
    unsigned int f = x.size(1);
    unsigned int w = x.size(2);
    unsigned int h = x.size(3);
    auto x_ = torch::permute(x, {1, 0, 2, 3}).contiguous().reshape({f, bs * w * h});

    auto rho = torch::empty_like(x_);
    auto rho_prime = torch::empty_like(x_);

    AT_DISPATCH_FLOATING_TYPES(scalar_type, "quartic_bspline_forward", [&] {
        quartic_bspline_forward_kernel<scalar_t><<<num_blocks, num_threads_per_block>>>(
            x_.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            weight_tensor.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(), 
            centers.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            static_cast<scalar_t>(scale),
            rho.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            rho_prime.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>()
        );
    });

    return {
        rho.reshape({f, bs, w, h}).permute({1, 0, 2, 3}).contiguous(), 
        rho_prime.reshape({f, bs, w, h}).permute({1, 0, 2, 3}).contiguous()
    };
}


void hello_cuda(torch::Tensor x){
    // C++ wrapper launching the above kernel
    int N = x.numel();
    int num_threads_per_block = 32;
    int num_blocks = (N + num_threads_per_block - 1) / num_threads_per_block;

    hello_kernel<<<num_blocks, num_threads_per_block>>>(x.data_ptr<float>(), N);

    cudaDeviceSynchronize();
}
