#include <torch/extension.h>

void hello_cuda(torch::Tensor x);

// Forward declaration
std::vector<torch::Tensor> quartic_bspline_forward_function(
    const torch::Tensor x, 
    const torch::Tensor weight_tensor,
    const torch::Tensor centers,
    const float scale
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("hello_cuda", &hello_cuda, "pybind11 example module");

    m.def("quartic_bspline_forward", &quartic_bspline_forward_function, "quartic bspline forward function");
}