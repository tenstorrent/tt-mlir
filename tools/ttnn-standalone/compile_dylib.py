#!/opt/ttmlir-toolchain/venv/bin/python
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-

import sys
import os
import subprocess
import shutil


def main():
    if len(sys.argv) != 3:
        print("Usage: script.py <path-to-cpp-file> <path-to-output-dir>")
        sys.exit(1)

    for name, value in os.environ.items():
        print("{0}: {1}".format(name, value))

    cpp_file_path = sys.argv[1]
    output_dir = sys.argv[2]

    # Verify the input file exists
    if not os.path.isfile(cpp_file_path):
        print(f"Error: File '{cpp_file_path}' does not exist.")
        sys.exit(1)

    # Verify the output directory exists
    if not os.path.isdir(output_dir):
        print(f"Error: Directory '{output_dir}' does not exist.")
        sys.exit(1)

    # Define the path to the target file
    tt_mlir_home = os.environ.get("TT_MLIR_HOME")
    if not tt_mlir_home:
        print("Error: TT_MLIR_HOME environment variable is not set.")
        sys.exit(1)

    target_file_path = os.path.join(
        tt_mlir_home, "tools/ttnn-standalone/ttnn-dylib.cpp"
    )

    try:
        # Read contents of the input file
        with open(cpp_file_path, "r") as source_file:
            cpp_content = source_file.read()

        # Overwrite the target file
        with open(target_file_path, "w") as target_file:
            target_file.write(cpp_content)

        print(
            f"Successfully updated {target_file_path} with contents from {cpp_file_path}."
        )
    except Exception as e:
        print(f"Error while handling files: {e}")
        sys.exit(1)

    # Define the commands to be executed
    build_dir = os.path.join(tt_mlir_home, "tools/ttnn-standalone/build")
    cmake_command = [
        "cmake",
        "-G",
        "Ninja",
        "-B",
        build_dir,
        "-S",
        os.path.join(tt_mlir_home, "tools/ttnn-standalone"),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_C_COMPILER=clang",
        "-DCMAKE_CXX_COMPILER=clang++",
    ]

    build_command = ["cmake", "--build", build_dir, "--", "ttnn-dylib"]

    try:
        # Run the cmake command
        print("Running cmake command...")
        subprocess.run(cmake_command, check=True, cwd=tt_mlir_home)

        # Run the build command
        print("Building ttnn-dylib...")
        subprocess.run(build_command, check=True, cwd=tt_mlir_home)

        print("Build completed successfully.")

        # Determine the output .so file
        compiled_so_path = os.path.join(build_dir, "libttnn-dylib.so")
        if not os.path.isfile(compiled_so_path):
            print(f"Error: Compiled file '{compiled_so_path}' not found.")
            sys.exit(1)

        # Define the destination path with renamed file
        output_file_name = os.path.basename(cpp_file_path)
        output_file_name = os.path.splitext(output_file_name)[0] + ".so"
        destination_path = os.path.join(output_dir, output_file_name)

        # Copy and rename the .so file
        shutil.copy2(compiled_so_path, destination_path)
        print(f"Successfully copied compiled file to {destination_path}.")
    except subprocess.CalledProcessError as e:
        print(f"Error during build process: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during file operations: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
