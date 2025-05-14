# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from setuptools import setup, find_packages

setup(
    name="ttir2torch",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A simple Python module",
    long_description="A longer description of your module",
    long_description_content_type="text/markdown",
    url="https://github.com/tenstorrent/tt-mlir",
    packages=["ttir2torch"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.6",
)
