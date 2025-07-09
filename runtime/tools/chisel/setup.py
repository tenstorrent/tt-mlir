# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from setuptools import setup, find_packages
import os

setup(
    name="chisel",
    version="0.1.0",
    packages=["chisel"],  # Explicitly list the package
    # find_packages() is also an option if you have multiple top-level packages
    # packages=find_packages(), # This would find 'tt_chisel' if it's a subdir with __init__.py
    install_requires=[
        "torch",
        # Add other dependencies like ttmlir if they are available on PyPI
        # If ttmlir or ttrt are local packages, they need to be installed separately.
    ],
    author="Nikola Drakulic",
    author_email="ndrakulic@tenstorrent.com",
    description="A chisel tool for MLIR processing.",
    # It's good practice to have a README.md for the long description
    long_description="",
    long_description_content_type="text/markdown",
    url="https://github.com/tenstorrent/tt-mlir",  # Replace with your project URL
)
