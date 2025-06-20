# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from setuptools import setup, find_packages

setup(
    name="tt_alchemist",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "click>=8.0.0",
        "pybind11>=2.6.0",
    ],
    entry_points={
        "console_scripts": [
            "tt-alchemist=tt_alchemist.cli:main",
        ],
    },
    author="Tenstorrent",
    author_email="info@tenstorrent.com",
    description="A user-friendly abstraction layer for tt-mlir",
    keywords="machine learning, compiler, tenstorrent",
    python_requires=">=3.6",
)
