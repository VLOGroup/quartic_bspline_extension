from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

extension_name = 'bspline_cuda_extension'
module = CUDAExtension(name=extension_name, 
                       sources=['bspline_cuda/bindings.cpp', 
                                'bspline_cuda/quartic_forward_kernel.cu', 
                                'bspline_cuda/quartic_backward_kernel.cu'])

module_path = os.path.dirname(__file__)
include_path = os.path.join(module_path, 'bspline_cuda', 'include')

package_name = 'bspline_cuda'
setup(name=package_name, 
      packages=[package_name], 
      package_dir={package_name: './bspline_cuda'}, 
      ext_package='bspline_cuda_extension', 
      ext_modules=[module], 
      cmdclass={'build_ext': BuildExtension},
      package_data={'bspline_cuda': ['include/*.cuh']})