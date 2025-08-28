#include <torch/extension.h>

#include "include/debug_utils.cuh"

std::vector<torch::Tensor> quartic_bspline_forward_function(
    const torch::Tensor x,
    const torch::Tensor weight_tensor,
    const torch::Tensor centers,
    const float scale);

std::vector<torch::Tensor> quartic_bspline_backward_function(
    const torch::Tensor x,
    const torch::Tensor weight_tensor,
    const torch::Tensor centers,
    const float scale,
    const torch::Tensor grad_out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("quartic_bspline_forward", &quartic_bspline_forward_function, 
          "quartic bspline forward function");
    m.def("quartic_bspline_backward", &quartic_bspline_backward_function,
          "quartic bspline backward function");
}