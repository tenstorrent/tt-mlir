# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Example script showing how to use the tt-alchemist API and CLI
"""

import os
import sys
import subprocess

# Import the direct API
from tt_alchemist import model_to_cpp


def api_example():
    """Example of using the Python API directly."""
    print("\n=== Using Direct Python API ===\n")

    # Path to the model file
    model_file = "/localdev/svuckovic/_workspace/repos/tt-mlir/tools/tt-alchemist/test/models/mnist.mlir"

    # Check if the file exists
    if not os.path.exists(model_file):
        print(f"Error: Input file '{model_file}' does not exist.")
        return False

    # Call the API function directly
    print(f"Converting {model_file} using direct API call...")
    result = model_to_cpp(model_file)

    if result:
        print("Successfully converted model to C++")
        return True
    else:
        print("Failed to convert model to C++")
        return False


def cli_example():
    """Example of using the CLI via subprocess."""
    print("\n=== Using CLI via Subprocess ===\n")

    # Path to the model file
    model_file = "/localdev/svuckovic/_workspace/repos/tt-mlir/tools/tt-alchemist/test/models/mnist.mlir"

    # Check if the file exists
    if not os.path.exists(model_file):
        print(f"Error: Input file '{model_file}' does not exist.")
        return 1

    # Run the CLI command
    cmd = [
        "python3",
        "-m",
        "tt_alchemist.cli",
        "convert",
        "--input",
        model_file,
        "--verbose",
    ]

    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("Conversion successful!")
        return 0
    else:
        print(f"Conversion failed with return code: {result.returncode}")
        return result.returncode


def main():
    # Run both examples
    api_result = api_example()
    cli_result = cli_example()

    # Return success only if both examples succeeded
    return 0 if api_result and cli_result == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
