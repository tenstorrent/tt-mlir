# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from setuptools import setup, find_packages

setup(
    name="tt-alchemist",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "click>=7.0",
    ],
    entry_points={
        "console_scripts": [
            "tt-alchemist=tt_alchemist.cli:cli",
        ],
    },
    author="Tenstorrent",
    description="Python CLI for tt-alchemist library",
)
