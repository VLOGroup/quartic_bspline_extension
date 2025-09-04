#include <torch/extension.h>

/**
 * @brief C++ wrapper of the CUDA kernel quartic_bspline_forward_kernel.
 * 
 * @param x Tensor of shape [bs, f, w, h]
 * @param weight_tensor Tensor of shape [f, num_centers] containing per feature/marginal
 *      and center node the weight of the corresponding shifted b-spline.
 * @param centers Tensor of shape [num_centers, ] containing the equally spaced
 *      centers nodes.
 * @param scale Scaling factor
 * @return std::vector<torch::Tensor> Container holding evaluation of spline potential
 *      at x and its derivative w.r.t. x evaluated at x.
 */
std::vector<torch::Tensor> quartic_bspline_forward_function(
    const torch::Tensor x,
    const torch::Tensor weight_tensor,
    const torch::Tensor centers,
    const double scale);

/**
 * @brief C++ wrapper of the CUDA kernel quartic_bspline_backward_kernel.
 * 
 * @param x Tensor of shape [bs, f, w, h]
 * @param weight_tensor Tensor of shape [f, num_centers] containing per feature/marginal
 *      and center node the weight of the corresponding shifted b-spline.
 * @param centers Tensor of shape [num_centers, ] containing the equally spaced
 *      centers nodes.
 * @param scale Scaling factor
 * @param grad_out Tensor of shape [bs, f, w, h] corresponding to the gradient 
 *      of (scalar) loss w.r.t. the output of spline potential.
 * @return std::vector<torch::Tensor> 
 */
std::vector<torch::Tensor> quartic_bspline_backward_function(
    const torch::Tensor x,
    const torch::Tensor weight_tensor,
    const torch::Tensor centers,
    const double scale,
    const torch::Tensor grad_out);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("quartic_bspline_forward", &quartic_bspline_forward_function, 
          "quartic bspline forward function");
    m.def("quartic_bspline_backward", &quartic_bspline_backward_function,
          "quartic bspline backward function");
}