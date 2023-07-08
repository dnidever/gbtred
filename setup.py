#!/usr/bin/env python

#from distutils.core import setup
from setuptools import setup, find_packages

setup(name="gbtred",
      version='1.0.0',
      description='GBT Data Reduction',
      author='David Nidever',
      author_email='dnidever@montana.edu',
      url='https://github.com/dnidever/gbtred',
      requires=['numpy','astropy(>=4.0)','scipy','dlnpyutils'],
      packages=find_namespace_packages(where="python"),
      package_dir={"": "python"} 
#      include_package_data=True,
)
