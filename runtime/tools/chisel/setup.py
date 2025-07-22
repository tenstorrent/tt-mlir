# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from setuptools import setup, find_packages
import os
from pathlib import Path

# Read the contents of README file if it exists
this_directory = Path(__file__).parent
readme_file = this_directory / "README.md"
long_description = ""
if readme_file.exists():
    long_description = readme_file.read_text(encoding="utf-8")

# Read version from __init__.py
version = "0.1.0"

setup(
    name="chisel",
    version=version,
    packages=["chisel", "chisel.core", "chisel.utils"],
    python_requires=">=3.8",
    install_requires=[
        "torch",
    ],
    author="Nikola Drakulic",
    author_email="ndrakulic@tenstorrent.com",
    description="A debugging and validation tool for TT-MLIR that compares MLIR operations between golden and device implementations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tenstorrent/tt-mlir",
    keywords="mlir, tenstorrent, debugging",
)
