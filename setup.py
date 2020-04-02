from setuptools import setup
from setuptools import find_packages
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy as np
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

ext_modules = [
    Extension(name="CTL.causal_tree.util_c", sources=["CTL/causal_tree/util_c.pyx"], include_dirs=[np.get_include()]),
    Extension(name="CTL.causal_tree.cython_ctl.ctl_base", sources=["CTL/causal_tree/cython_ctl/ctl_base.pyx"],
              include_dirs=[np.get_include()])
]

setuptools.setup(
    name="causal_tree_learn",
    version="2.2",
    author="Christopher Tran",
    author_email="ctran29@uic.edu",
    description="Python implementation of causal trees with validation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/edgeslab/CTL",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['numpy',
                      'scikit-learn',
                      'scipy'
                      ],
    python_requires='>=3.6',
    ext_modules=cythonize(ext_modules),
    cmdclass={'build_ext': build_ext}
)
