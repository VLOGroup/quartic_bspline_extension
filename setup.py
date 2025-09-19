from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
import torch
import os

extension_name = 'bspline_extension'
if torch.cuda.is_available():
    module = CUDAExtension(name=extension_name, 
                           sources=['bspline_cuda/bindings.cpp', 
                                    'bspline/quartic_forward_cuda_kernel.cu', 
                                    'bspline/quartic_backward_cuda_kernel.cu',
                                    'bspline/quartic_forward_cpu_kernel.cpp',
                                    'bspline/quartic_backward_cpu_kernel.cpp'])

    module_path = os.path.dirname(__file__)
    include_path = os.path.join(module_path, 'bspline', 'include')
else:
    module = CppExtension(name=extension_name, 
                           sources=['bspline/bindings.cpp',
                                    'bspline/quartic_forward_cpu_kernel.cpp',
                                    'bspline/quartic_backward_cpu_kernel.cpp'])

    module_path = os.path.dirname(__file__)
    include_path = os.path.join(module_path, 'bspline', 'include')
    



package_name = 'bspline'
setup(name=package_name, 
      packages=[package_name], 
      package_dir={package_name: './bspline'}, 
      ext_package=extension_name, 
      ext_modules=[module], 
      cmdclass={'build_ext': BuildExtension},
      package_data={'bspline': ['include/*.cuh']})