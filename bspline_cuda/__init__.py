from torch.utils.cpp_extension import load
import os

module_path = os.path.dirname(__file__)
quartic_bspline_cuda_extension = load(name='quartic_bspline_cuda_extension', 
                              sources=[os.path.join(module_path, 'bindings.cpp'),
                              os.path.join(module_path, 'kernel.cu')], verbose=True)
quartic_bspline_forward = quartic_bspline_cuda_extension.quartic_bspline_forward
