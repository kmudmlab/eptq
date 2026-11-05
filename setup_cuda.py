"""
Build the EPTQ CUDA extension (ahead-of-time compilation).

Usage:
    python setup_cuda.py build_ext --inplace

The resulting _eptq_cuda*.so will be found automatically by kernels/__init__.py.
JIT compilation via torch.utils.cpp_extension.load() is also supported as a
fallback; this script is only needed for faster startup in production.
"""

import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CSRC = os.path.join(os.path.dirname(__file__), "csrc")

setup(
    name="_eptq_cuda",
    ext_modules=[
        CUDAExtension(
            name="_eptq_cuda",
            sources=[
                os.path.join(CSRC, "eptq_kernels.cu"),
                os.path.join(CSRC, "binding.cpp"),
            ],
            include_dirs=[CSRC],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "--expt-relaxed-constexpr",
                    "-std=c++17",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
