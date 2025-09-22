from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
import torch
import os

extension_name = 'quartic_bspline_extension'
if torch.cuda.is_available():
    module = CUDAExtension(name=extension_name, 
                           sources=['quartic_bspline_extension/bindings.cpp', 
                                    'quartic_bspline_extension/quartic_forward_cuda_kernel.cu', 
                                    'quartic_bspline_extension/quartic_backward_cuda_kernel.cu',
                                    'quartic_bspline_extension/quartic_forward_cpu_kernel.cpp',
                                    'quartic_bspline_extension/quartic_backward_cpu_kernel.cpp'],
                           extra_compile_args={'cxx': ['-fopenmp']},
                           extra_link_args=['-fopenmp'])

    module_path = os.path.dirname(__file__)
    include_path = os.path.join(module_path, 'quartic_bspline_extension', 'include')
else:
    module = CppExtension(name=extension_name, 
                           sources=['quartic_bspline_extension/bindings.cpp',
                                    'quartic_bspline_extension/quartic_forward_cpu_kernel.cpp',
                                    'quartic_bspline_extension/quartic_backward_cpu_kernel.cpp'],
                          extra_compile_args={'cxx': ['-fopenmp'], 'nvcc': []},
                          extra_link_args=['-fopenmp'])

    module_path = os.path.dirname(__file__)
    include_path = os.path.join(module_path, 'quartic_bspline_extension', 'include')
    



package_name = 'quartic_bspline_extension'
setup(name=package_name, 
      packages=[package_name], 
      package_dir={package_name: './quartic_bspline_extension'}, 
      ext_package=extension_name, 
      ext_modules=[module], 
      cmdclass={'build_ext': BuildExtension},
      package_data={'quartic_bspline_extension': ['include/*.cuh', 'include/*.h']}, 
      python_requires='>=3.11')