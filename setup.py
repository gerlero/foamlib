"""Setup script for building C extension modules."""

from setuptools import Extension, setup

# Define the C extension module
skip_ext = Extension(
    "foamlib._files._parsing._skip_ext",
    sources=["src/foamlib/_files/_parsing/_skip_ext.c"],
)

setup(
    ext_modules=[skip_ext],
)
