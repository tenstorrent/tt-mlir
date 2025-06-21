#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Command-line interface for tt-alchemist library."""

import os
import sys
import click
from pathlib import Path

# Import the API functions
from .api import model_to_cpp


@click.group()
def cli():
    """TT-Alchemist - Model conversion and optimization tool."""
    pass


@cli.command()
@click.option(
    "--input",
    "-i",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="Input MLIR model file",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def convert(input, verbose):
    """Convert MLIR model to C++ code."""
    try:
        # Get absolute path
        input_path = os.path.abspath(input)

        if verbose:
            click.echo(f"Converting {input_path} to C++")

        # Call the API function
        result = model_to_cpp(input_path)

        # if result:
        #     click.echo(f"Successfully converted model to C++")
        #     return 0
        # else:
        #     click.echo("Failed to convert model to C++", err=True)
        #     return 1

    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        return 1


if __name__ == "__main__":
    sys.exit(cli())
