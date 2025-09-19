__version__ = '0.2.2'

from torch.utils.cpp_extension import load
import os

module_path = os.path.dirname(__file__)

quartic_bspline_extension = load(name='quartic_bspline__extension', 
                                 sources=[os.path.join(module_path, 'bindings.cpp'),
                                          os.path.join(module_path, 'quartic_forward_cuda_kernel.cu'),
                                          os.path.join(module_path, 'quartic_backward_cuda_kernel.cu'),
                                          os.path.join(module_path, 'quartic_forward_cpu_kernel.cpp'),
                                          os.path.join(module_path, 'quartic_backward_cpp_kernel.cpp')], 
                                      verbose=True)
quartic_bspline_forward = quartic_bspline_extension.quartic_bspline_forward
quartic_bspline_backward = quartic_bspline_extension.quartic_bspline_backward
