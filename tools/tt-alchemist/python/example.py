#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Example script showing how to use the tt-alchemist CLI directly
"""

import os
import sys
import subprocess


def main():
    # Path to the model file
    model_file = "/localdev/svuckovic/_workspace/repos/tt-mlir/tools/tt-alchemist/test/models/mnist.mlir"

    # Output directory
    output_dir = "./output"

    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Run the CLI command
    cmd = [
        "python3",
        f"{os.environ['TT_MLIR_HOME']}/tools/tt-alchemist/python/tt_alchemist_cli.py",
        "convert",
        "--input",
        model_file,
        "--output-dir",
        output_dir,
        "--verbose",
    ]

    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("Conversion successful!")
    else:
        print(f"Conversion failed with return code: {result.returncode}")


if __name__ == "__main__":
    main()
