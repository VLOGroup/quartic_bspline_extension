from torch.utils.cpp_extension import load
import os

module_path = os.path.dirname(__file__)

# NOTE
# ----
#   > package name and name of compiled extension cannot have the same name - leads to 
#       import issues!

quartic_bspline_extension = load(name='_quartic_bspline_extension', 
                                 sources=[os.path.join(module_path, 'bindings.cpp'),
                                          os.path.join(module_path, 'quartic_forward_cuda_kernel.cu'),
                                          os.path.join(module_path, 'quartic_backward_cuda_kernel.cu'),
                                          os.path.join(module_path, 'quartic_forward_cpu_kernel.cpp'),
                                          os.path.join(module_path, 'quartic_backward_cpu_kernel.cpp')], 
                                      verbose=True)
quartic_bspline_forward = quartic_bspline_extension.quartic_bspline_forward
quartic_bspline_backward = quartic_bspline_extension.quartic_bspline_backward

