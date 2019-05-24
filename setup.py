from setuptools import setup
from setuptools import find_packages

setup(name='ctl',
      version='1.0',
      description='Python implementation of causal trees with validation',
      author='Christopher Tran',
      author_email='ctran29@uic.edu',
      url='https://www.cs.uic.edu/~ctran/',
      download_url='https://github.com/chris-tran-16/CTL',
      install_requires=['numpy',
                        'scikit-learn',
                        'scipy'
                        ],
      package_data={'CTL': ['README.md', 'CTL/data/']},
      packages=find_packages())
