# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Command-line interface for tt-alchemist library."""

import os
import sys
import click
from pathlib import Path

# Import the API functions
from tt_alchemist.api import (
    model_to_cpp,
    model_to_python,
    generate_cpp,
    generate_python,
)


@click.group()
def cli():
    """tt-alchemist - Model conversion and optimization tool."""
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

    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        return 1


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def model_to_python_cmd(input_file, verbose):
    """Convert MLIR model to Python code."""
    try:
        if verbose:
            click.echo(f"Converting {input_file} to Python")

        success = model_to_python(input_file)

    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        return 1


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    "output_dir",
    type=click.Path(),
    required=True,
    help="Output directory path",
)
@click.option(
    "--local",
    "mode",
    flag_value="local",
    default=True,
    help="Generate for local execution (default)",
)
@click.option(
    "--standalone",
    "mode",
    flag_value="standalone",
    help="Generate for standalone execution",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def generate_cpp_cmd(input_file, output_dir, mode, verbose):
    """Generate standalone solution with the generated C++ code.

    This generates a directory with all necessary files to build and run the
    generated code, including CMakeLists.txt, precompiled headers, and a main
    C++ file.
    """
    try:
        if verbose:
            click.echo(
                f"Generating {mode} solution from {input_file} in directory {output_dir}"
            )

        is_local = mode == "local"
        success = generate_cpp(input_file, output_dir, is_local)
        if success:
            click.echo(f"Successfully generated {mode} solution in: {output_dir}")
            return 0
        else:
            click.echo(f"Failed to generate {mode} solution in: {output_dir}")
            return 1
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        return 1


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    "output_dir",
    type=click.Path(),
    required=True,
    help="Output directory path",
)
@click.option(
    "--local",
    "mode",
    flag_value="local",
    default=True,
    help="Generate for local execution (default)",
)
@click.option(
    "--standalone",
    "mode",
    flag_value="standalone",
    help="Generate for standalone execution",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def generate_python_cmd(input_file, output_dir, mode, verbose):
    """Generate standalone solution with the generated Python code.

    This generates a directory with all necessary files to build and run the
    generated code, including CMakeLists.txt, precompiled headers, and a main
    Python file.
    """
    try:
        if verbose:
            click.echo(
                f"Generating {mode} solution from {input_file} in directory {output_dir}"
            )

        is_local = mode == "local"
        success = generate_python(input_file, output_dir, is_local)
        if success:
            click.echo(f"Successfully generated {mode} solution in: {output_dir}")
            return 0
        else:
            click.echo(f"Failed to generate {mode} solution in: {output_dir}")
            return 1
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        return 1


if __name__ == "__main__":
    sys.exit(cli())
