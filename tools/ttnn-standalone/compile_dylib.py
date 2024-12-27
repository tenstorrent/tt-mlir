#!/opt/ttmlir-toolchain/venv/bin/python
# -*- coding: utf-8 -*-
#
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import sys
import os
import subprocess
import shutil


def main():
    if len(sys.argv) != 3:
        print("Usage: script.py <path-to-cpp-file> <path-to-output-dir>")
        sys.exit(1)

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

    # Define the path to the target
    tt_mlir_home = os.environ.get("TT_MLIR_HOME")
    if not tt_mlir_home:
        print("Error: TT_MLIR_HOME environment variable is not set.")
        sys.exit(1)

    # Create unique source and build directories
    cpp_base_name = os.path.basename(cpp_file_path).split(".")[0]
    temp_source_dir = os.path.join(
        tt_mlir_home, f"tools/ttnn-standalone/temp_source_{cpp_base_name}"
    )
    temp_build_dir = os.path.join(
        tt_mlir_home, f"tools/ttnn-standalone/temp_build_{cpp_base_name}"
    )

    try:
        # Create the temporary source directory and copy necessary files
        os.makedirs(temp_source_dir, exist_ok=True)
        shutil.copy2(
            os.path.join(tt_mlir_home, "tools/ttnn-standalone/CMakeLists.txt"),
            temp_source_dir,
        )
        shutil.copy2(
            os.path.join(tt_mlir_home, "tools/ttnn-standalone/ttnn-precompiled.hpp"),
            temp_source_dir,
        )
        shutil.copy2(
            os.path.join(tt_mlir_home, "tools/ttnn-standalone/ttnn-dylib.hpp"),
            temp_source_dir,
        )
        shutil.copy2(
            os.path.join(tt_mlir_home, "tools/ttnn-standalone/ttnn-standalone.cpp"),
            temp_source_dir,
        )  # TODO(svuckovic): remove the need for this

        # Copy the input .cpp file to the temporary source directory
        temp_cpp_path = os.path.join(temp_source_dir, "ttnn-dylib.cpp")
        shutil.copy2(cpp_file_path, temp_cpp_path)

        print(f"Temporary source directory created: {temp_source_dir}")

        # Define the commands to be executed
        cmake_command = [
            "cmake",
            "-G",
            "Ninja",
            "-B",
            temp_build_dir,
            "-S",
            temp_source_dir,
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_C_COMPILER=clang",
            "-DCMAKE_CXX_COMPILER=clang++",
        ]

        build_command = ["cmake", "--build", temp_build_dir, "--", "ttnn-dylib"]

        # Run the cmake command
        print("Running cmake command...")
        subprocess.run(cmake_command, check=True, cwd=tt_mlir_home)

        # Run the build command
        print("Building ttnn-dylib...")
        subprocess.run(build_command, check=True, cwd=tt_mlir_home)

        print("Build completed successfully.")

        # Determine the output .so file
        compiled_so_path = os.path.join(temp_build_dir, "libttnn-dylib.so")
        if not os.path.isfile(compiled_so_path):
            print(f"Error: Compiled file '{compiled_so_path}' not found.")
            sys.exit(1)

        # Define the destination path with renamed file
        output_file_name = cpp_base_name + ".so"
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
    finally:
        # Cleanup temporary directories
        shutil.rmtree(temp_source_dir, ignore_errors=True)
        shutil.rmtree(temp_build_dir, ignore_errors=True)
        print(
            f"Cleaned up temporary directories: {temp_source_dir} and {temp_build_dir}"
        )


if __name__ == "__main__":
    main()
