from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='quartic_bspline_cuda_extension',
    ext_modules=[
        CUDAExtension(
            name='quartic_bspline_cuda_extension', 
            sources=['bspline_cuda/bindings.cpp', 'bspline_cuda/kernel.cu'])
    ],
    cmdclass={'build_ext': BuildExtension}
)