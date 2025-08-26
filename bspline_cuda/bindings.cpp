#include <torch/extension.h>

#include "include/debug_utils.cuh"

std::vector<torch::Tensor> quartic_bspline_forward_function(
    torch::Tensor x,
    torch::Tensor weight_tensor,
    torch::Tensor centers,
    float scale);


    
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("quartic_bspline_forward", &quartic_bspline_forward_function, "quartic bspline forward function");
}