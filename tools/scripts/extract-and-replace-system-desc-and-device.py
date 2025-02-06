# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
This script is used to extract and replace system and device descriptions in an MLIR
file. It is useful for testing purposes where system and device descriptions are not
known at the time of writing the test. The script generates the system and device
descriptions using the ttmlir-opt tool and replaces the existing system and device
descriptions in the input MLIR file with the generated descriptions.

Usage:
    The script is executed with three command-line arguments:
    --input-file: Path to the input file to replace system and device descriptions.
    --temp-directory: Path to the temporary execution directory.
    --system-desc-path: Path to the system description file.
"""

import sys
import subprocess
import os
import argparse


def generate_system_desc_and_device_mlir(temp_dir, system_desc_path):
    # Write an empty module to temp_dir/temp_empty_module.mlir
    temp_empty_module_path = os.path.join(temp_dir, "temp_empty_module.mlir")
    with open(temp_empty_module_path, "w") as temp_file:
        temp_file.write("module {}")

    # Construct the command to run ttmlir-opt tool
    command = (
        f'ttmlir-opt --ttir-load-system-desc="path={system_desc_path}" '
        f"--ttir-implicit-device {temp_empty_module_path}"
    )

    # Execute the command and capture the output
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    # Check if the command was successful
    if result.returncode != 0:
        print("Error running the command")
        print("stdout:", result.stdout)
        print("stderr:", result.stderr)
        sys.exit(1)

    return result.stdout


def extractAndReplaceSystemDescAndDeviceDesc(input_file, sytem_and_device_desc_mlir):
    # Split the input mlir content into lines
    input_lines = sytem_and_device_desc_mlir.splitlines()

    system_desc = ""
    device_desc = ""
    # Read content from the input file
    for line in input_lines:
        if "#system_desc =" in line:
            system_desc = line.strip()
        if "#device =" in line:
            device_desc = line.strip()

    if system_desc == "":
        print("System description not found in the input mlir!")
        sys.exit(1)

    if device_desc == "":
        print("Device description not found in the input mlir!")
        sys.exit(1)

    # Write the modified content with replaced values
    modified_content = ""
    with open(input_file, "r") as file:
        for line in file:
            if line.strip().startswith("#device ="):
                modified_content += device_desc + "\n"
            elif line.strip().startswith("#system_desc ="):
                modified_content += system_desc + "\n"
            elif line.strip().startswith("// RUN:"):
                continue
            else:
                modified_content += line

    # Write the modified content to the stdout
    print(modified_content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract and replace system and device descriptions."
    )
    parser.add_argument(
        "--input-file",
        help="Path to the input file to replace system and device descriptions.",
    )
    parser.add_argument(
        "--temp-directory", help="Path to the temp execution directory."
    )
    parser.add_argument(
        "--system-desc-path", help="Path to the system description file."
    )

    args = parser.parse_args()

    temp_dir = args.temp_directory
    system_desc_path = args.system_desc_path
    input_file = args.input_file

    if input_file is None:
        print(
            "Input file path for system and device descriptor replacement is not provided! Exiting..."
        )
        sys.exit(1)

    if not os.path.exists(input_file):
        print(
            f"Input file {input_file} for system and device descriptor replacement does not exist! Exiting..."
        )
        sys.exit(1)

    if not os.path.exists(temp_dir):
        print(f"Temp directory {temp_dir} does not exist! Exiting...")
        sys.exit(1)

    generated_system_and_device_desc_mlir = generate_system_desc_and_device_mlir(
        temp_dir, system_desc_path
    )

    extractAndReplaceSystemDescAndDeviceDesc(
        input_file, generated_system_and_device_desc_mlir
    )
