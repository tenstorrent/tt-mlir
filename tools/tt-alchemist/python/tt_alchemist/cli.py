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
from tt_alchemist.api import model_to_cpp, generate


@click.group()
def cli():
    """TT-Alchemist - Model conversion and optimization tool."""
    pass


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def model_to_cpp_cmd(input_file, verbose):
    """Convert MLIR model to C++ code."""
    try:
        if verbose:
            click.echo(f"Converting {input_file} to C++")

        success = model_to_cpp(input_file)

        # if success:
        #     click.echo("Conversion successful!")
        #     return 0
        # else:
        #     click.echo("Conversion failed!")
        #     return 1
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        return 1


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path())
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def generate_cmd(input_file, output_dir, verbose):
    """Generate a standalone solution with the generated C++ code.

    This generates a directory with all necessary files to build and run the
    generated code, including CMakeLists.txt, precompiled headers, and a main
    C++ file.
    """
    try:
        if verbose:
            click.echo(
                f"Generating solution from {input_file} in directory {output_dir}"
            )

        success = generate(input_file, output_dir)
        if success:
            click.echo(f"Successfully generated solution in: {output_dir}")
            return 0
        else:
            click.echo(f"Failed to generate solution in: {output_dir}")
            return 1
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        return 1


if __name__ == "__main__":
    sys.exit(cli())
