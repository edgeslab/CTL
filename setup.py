from setuptools import setup
from setuptools import find_packages
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="causal_tree_learn",
    version="1.0.17",
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
)
