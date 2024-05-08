import os
from setuptools import setup, find_packages
import torch
from torch.utils import cpp_extension

compute_capability = torch.cuda.get_device_capability()
cuda_arch = compute_capability[0] * 100 + compute_capability[1] * 10

setup(
    name='cuda_tools',
    ext_modules=[
        cpp_extension.CUDAExtension(
            name='cuda_tools.attn',
            sources=[
                'attn_tools.cu',
                "binding.cpp"
            ],
            extra_link_args=['-lcublas_static', '-lcublasLt_static',
                             '-lculibos', '-lcudart', '-lcudart_static',
                             '-lrt', '-lpthread', '-ldl', '-L/usr/lib/x86_64-linux-gnu/'],
            extra_compile_args={'cxx': ['-std=c++17', '-O3'],
                                'nvcc': ['-O3', '-std=c++17', '-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_HALF2_OPERATORS__', f'-DCUDA_ARCH={cuda_arch}']},
        ),
    ],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension.with_options(use_ninja=True)
    },
    packages=find_packages(
        exclude=['notebook', 'scripts', 'tests']),
)
