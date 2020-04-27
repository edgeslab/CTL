# from setuptools import setup
from setuptools import find_packages
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy as np

try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

with open("README.md", "r") as fh:
    long_description = fh.read()

cmdclass = {}
ext_modules = []

if use_cython:
    ext_modules = [
        Extension(name="CTL.causal_tree.util_c", sources=["CTL/causal_tree/util_c.pyx"],
                  include_dirs=[np.get_include(), "."]),
    ]
    cmdclass.update({'build_ext': build_ext})
else:
    # ext_modules = [
    #     Extension(name="CTL.causal_tree.util_c", sources=["CTL/causal_tree/util_c.pyx",  "CTL/causal_tree/util_c.c"],
    #               include_dirs=[np.get_include(), "."]),
    # ]
    ext_modules = [
        Extension(name="CTL.causal_tree.util_c", sources=["CTL/causal_tree/util_c.c"],
                  include_dirs=[np.get_include(), "."]),
    ]


setup(
    name="causal_tree_learn",
    version="2.30",
    author="Christopher Tran",
    author_email="ctran29@uic.edu",
    description="Python implementation of causal trees with validation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/edgeslab/CTL",
    packages=find_packages(),
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
    # cmdclass={'build_ext': build_ext},
    cmdclass=cmdclass,
    setup_requires=["cython", "numpy"],
    package_data={"CTL.causal_tree": ["util_c.c", "util_c.pyx"]}
)
