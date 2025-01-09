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


def get_emitc_tests_path():
    if get_emitc_tests_path.path:
        return get_emitc_tests_path.path

    tt_mlir_home = os.environ.get("TT_MLIR_HOME")
    if not tt_mlir_home:
        print("Error: TT_MLIR_HOME environment variable is not set.")
        sys.exit(1)

    get_emitc_tests_path.path = os.path.join(
        tt_mlir_home, "build/test/ttmlir/EmitC/TTNN"
    )

    return get_emitc_tests_path.path


get_emitc_tests_path.path = None


def compile_shared_object(cpp_file_path, output_dir):
    tt_mlir_home = os.environ.get("TT_MLIR_HOME")
    if not tt_mlir_home:
        print("Error: TT_MLIR_HOME environment variable is not set.")
        sys.exit(1)

    cpp_base_name = os.path.basename(cpp_file_path).rsplit(".", 1)[0]
    temp_source_dir = os.path.join(
        tt_mlir_home, f"tools/ttnn-standalone/temp_source_{cpp_base_name}"
    )
    temp_build_dir = os.path.join(
        tt_mlir_home, f"tools/ttnn-standalone/temp_build_{cpp_base_name}"
    )

    try:
        os.makedirs(temp_source_dir, exist_ok=True)
        for file in [
            "CMakeLists.txt",
            "ttnn-precompiled.hpp",
            "ttnn-standalone.cpp",
        ]:
            shutil.copy2(
                os.path.join(tt_mlir_home, f"tools/ttnn-standalone/{file}"),
                temp_source_dir,
            )

        temp_cpp_path = os.path.join(temp_source_dir, "ttnn-dylib.cpp")
        shutil.copy2(cpp_file_path, temp_cpp_path)

        cmake_command = [
            "cmake",
            "-G",
            "Ninja",
            "-B",
            temp_build_dir,
            "-S",
            temp_source_dir,
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_CXX_COMPILER=clang++",
        ]
        build_command = ["cmake", "--build", temp_build_dir, "--", "ttnn-dylib"]

        subprocess.run(cmake_command, check=True, cwd=tt_mlir_home)
        subprocess.run(build_command, check=True, cwd=tt_mlir_home)

        compiled_so_path = os.path.join(temp_build_dir, "libttnn-dylib.so")
        if not os.path.isfile(compiled_so_path):
            print(f"Error: Compiled file '{compiled_so_path}' not found.")
            sys.exit(1)

        output_file_name = cpp_base_name + ".so"
        destination_path = os.path.join(output_dir, output_file_name)
        shutil.copy2(compiled_so_path, destination_path)
        print(f"Successfully copied compiled file to {destination_path}.")
    except subprocess.CalledProcessError as e:
        print(f"Error during build process: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during file operations: {e}")
        sys.exit(1)
    finally:
        shutil.rmtree(temp_source_dir, ignore_errors=True)
        shutil.rmtree(temp_build_dir, ignore_errors=True)
        print(
            f"Cleaned up temporary directories: {temp_source_dir} and {temp_build_dir}"
        )


def main():
    if len(sys.argv) != 1:
        print("Usage: ci_compile_dylib.py")
        sys.exit(1)

    test_path = get_emitc_tests_path()

    if not os.path.isdir(test_path):
        print(f"Error: Directory '{test_path}' does not exist.")
        sys.exit(1)

    for root, _, files in os.walk(test_path):
        for file in files:
            if file.endswith(".cpp"):
                cpp_file_path = os.path.join(root, file)
                compile_shared_object(cpp_file_path, test_path)


if __name__ == "__main__":
    main()
