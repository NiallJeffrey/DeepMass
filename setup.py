#!/usr/bin/env python

from setuptools import setup, find_packages
import sys

setup(name='deepmass',
      version='1.0',
      description='Deep learning mass maps',
      author='Niall Jeffrey',
      url='https://github.com/NiallJeffrey/DeepMass',
      packages=find_packages(),
      install_requires=[
          "tensorflow>=2.2.0",
          "healpy"
      ])

