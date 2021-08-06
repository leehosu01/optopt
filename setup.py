#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 16:04:10 2021

@author: map
"""
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open("README.md", "r") as fh:
    long_description = fh.read()

import glob
def recursive_dir_walker(dir):
    files = [recursive_dir_walker(directory) for directory in glob.glob(f'{dir}/*')]
    if len(files) == 0: files = [[dir]]
    return sum(files, [])

datafiles = []#recursive_dir_walker("ABC/*")

setuptools.setup(
    name="optimizer optimizer",
    version="0.0.0-alpha",
    author="Hosu Lee",
    author_email="leehosu01@naver.com",
    description="Control Optimizer's hyperparameter dynamically",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/leehosu01/optopt",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=requirements,
    data_files = datafiles,
    include_package_data=True
)
