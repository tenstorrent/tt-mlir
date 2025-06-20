#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TT-Alchemist CLI - Python wrapper for the tt-alchemist library
"""

import os
import sys
import click
import ctypes
from pathlib import Path

# Load the tt-alchemist shared library
def load_library():
    # Adjust this path based on where the library is installed
    lib_path = os.environ.get("TT_ALCHEMIST_LIB_PATH", None)
    if not lib_path:
        # Try to find the library in common locations
        possible_paths = [
            # Add paths relative to this script
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "../../../build/lib/libtt-alchemist.so",
            ),
            "/usr/local/lib/libtt-alchemist.so",
            "/usr/lib/libtt-alchemist.so",
        ]

        for path in possible_paths:
            if os.path.exists(path):
                lib_path = path
                break

    if not lib_path or not os.path.exists(lib_path):
        raise RuntimeError(
            "Could not find tt-alchemist library. Set TT_ALCHEMIST_LIB_PATH environment variable."
        )

    try:
        print(f"loading alchemist lib: {lib_path}")
        return ctypes.CDLL(lib_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load tt-alchemist library: {e}")


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
@click.option(
    "--output-dir",
    "-o",
    required=True,
    type=click.Path(file_okay=False, dir_okay=True),
    help="Output directory for generated C++ code",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def convert(input, output_dir, verbose):
    """Convert MLIR model to C++ code."""
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Get absolute paths
        input_path = os.path.abspath(input)
        output_dir_path = os.path.abspath(output_dir)

        if verbose:
            click.echo(f"Converting {input_path} to C++ in {output_dir_path}")

        # Load the library
        lib = load_library()
        print(f"Loaded library: {lib}")

        # Get the singleton instance
        lib.tt_alchemist_TTAlchemist_getInstance.restype = ctypes.c_void_p
        instance = lib.tt_alchemist_TTAlchemist_getInstance()

        # Set up function argument types
        lib.tt_alchemist_TTAlchemist_modelToCpp.argtypes = [
            ctypes.c_void_p,  # instance pointer
            ctypes.c_char_p,  # input_file
            ctypes.c_char_p,  # output_dir
        ]
        lib.tt_alchemist_TTAlchemist_modelToCpp.restype = ctypes.c_bool

        # Call the modelToCpp function
        result = lib.tt_alchemist_TTAlchemist_modelToCpp(
            instance, input_path.encode("utf-8"), output_dir_path.encode("utf-8")
        )

        if result:
            click.echo(f"Successfully converted model to C++")
            return 0
        else:
            click.echo("Failed to convert model to C++", err=True)
            return 1

    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        return 1


if __name__ == "__main__":
    sys.exit(cli())
