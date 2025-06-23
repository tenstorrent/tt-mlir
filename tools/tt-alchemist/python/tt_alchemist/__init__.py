# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TT-Alchemist Python package."""

# Import the CLI function to make it available when importing the package
from .cli import cli

# Import and expose the API functions
from .api import model_to_cpp, create_solution
