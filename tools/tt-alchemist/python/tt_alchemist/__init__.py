# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TT-Alchemist Python package."""

# Set up the library path before importing anything else
from .lib_setup import setup_library_path

setup_library_path()

# Import the CLI function to make it available when importing the package
from .cli import cli

# Import and expose the API functions
from .api import model_to_cpp, generate
