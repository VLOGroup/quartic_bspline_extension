from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

name = 'bspline_cuda_extension'
module = CUDAExtension(name=name, 
                       sources=['bspline_cuda/bindings.cpp', 
                                'bspline_cuda/quartic_forward_kernel.cu', 
                                'bspline_cuda/quartic_backward_kernel.cu'])

setup(name=name, ext_modules=[module], cmdclass={'build_ext': BuildExtension})