#!/usr/bin/env python

from setuptools import setup, find_packages
import sys

setup(name='deepmass',
      version='v0.1',
      description='Deep learning mass maps',
      author='Niall Jeffrey',
      url='https://github.com/NiallJeffrey/DeepMass',
      packages=find_packages(),
      install_requires=[
          "keras>=v2.2.4",
          "numpy"
      ])

