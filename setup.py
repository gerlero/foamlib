"""Setup script for foamlib with C extension."""

from setuptools import setup, Extension
import numpy as np

# Define the C extension
parse_ascii_ext = Extension(
    name="foamlib._files._parsing._parse_ascii",
    sources=["src/foamlib/_files/_parsing/_c_ext/parse_ascii.c"],
    include_dirs=[np.get_include()],
    extra_compile_args=["-O3", "-Wall"],
)

# Setup
setup(
    ext_modules=[parse_ascii_ext],
)
