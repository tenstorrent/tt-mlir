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


# Returns the path where EmitC TTNN tests live
#
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


# Returns TT_MLIR_HOME env var value
#
def get_ttmlir_home():
    tt_mlir_home = os.environ.get("TT_MLIR_HOME")
    if not tt_mlir_home:
        print("Error: TT_MLIR_HOME environment variable is not set.")
        sys.exit(1)

    return tt_mlir_home


# Returns ttnn-standalone dir
#
def get_standalone_dir():
    return os.path.join(get_ttmlir_home(), "tools/ttnn-standalone")


# Runs cmake setup for .so compilation
# Runs only once per script
#
def run_cmake_setup():
    if run_cmake_setup.already_created:
        return

    tt_mlir_home = get_ttmlir_home()

    source_dir = get_standalone_dir()
    build_dir = os.path.join(source_dir, "build")

    cmake_command = [
        "cmake",
        "-G",
        "Ninja",
        "-B",
        build_dir,
        "-S",
        source_dir,
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CXX_COMPILER=clang++",
    ]

    try:
        result = subprocess.run(
            cmake_command, check=True, cwd=tt_mlir_home, capture_output=True, text=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error setting up cmake environment: {e}")
        print(e.stderr)
        print(e.stdout)
        sys.exit(1)
    except Exception as e:
        print(f"Error setting up cmake environment: {e}")
        sys.exit(1)

    run_cmake_setup.already_created = True


# Init fn variables
#
get_emitc_tests_path.path = None
run_cmake_setup.already_created = None


# Compile shared object, given source cpp and dest dir
#
def compile_shared_object(cpp_file_path, output_dir):
    tt_mlir_home = os.environ.get("TT_MLIR_HOME")
    if not tt_mlir_home:
        print("Error: TT_MLIR_HOME environment variable is not set.")
        sys.exit(1)

    # Base name of the provided cpp file
    #
    cpp_base_name = os.path.basename(cpp_file_path).rsplit(".", 1)[0]

    # Various dirs
    #
    source_dir = get_standalone_dir()
    build_dir = os.path.join(source_dir, "build")
    source_cpp_path = os.path.join(source_dir, "ttnn-dylib.cpp")
    compiled_so_path = os.path.join(build_dir, "libttnn-dylib.so")

    try:
        # Copy provided cpp file to source dir
        #
        shutil.copy(cpp_file_path, source_cpp_path)

        # Run cmake setup command first
        #
        run_cmake_setup()

        # Remove previous .so if exists
        if os.path.exists(compiled_so_path):
            os.remove(compiled_so_path)

        # Run build
        #
        print(f"\nBuilding: {cpp_base_name}")
        build_command = ["cmake", "--build", build_dir, "--", "ttnn-dylib"]
        result = subprocess.run(
            build_command, check=True, cwd=tt_mlir_home, capture_output=True, text=True
        )
        print(f"  Build finished successfully!")

        # Confirm .so exists
        if not os.path.isfile(compiled_so_path):
            print(f"Error: Compiled file '{compiled_so_path}' not found.")
            sys.exit(1)

        # Copy the compiled .so
        #
        output_file_name = cpp_base_name + ".so"
        destination_path = os.path.join(output_dir, output_file_name)
        shutil.copy2(compiled_so_path, destination_path)
        print(f"  Successfully copied compiled file to {destination_path}.")
    except subprocess.CalledProcessError as e:
        print(f"  Error during build process: {e}")
        print(e.stderr)
        print(e.stdout)
        sys.exit(1)
    except Exception as e:
        print(f"  Error during file operations: {e}")
        print(e.stderr)
        print(e.stdout)
        sys.exit(1)
    finally:
        pass


def main():
    if len(sys.argv) != 1:
        print("Usage: ci_compile_dylib.py")
        sys.exit(1)

    test_path = get_emitc_tests_path()

    if not os.path.isdir(test_path):
        print(f"Error: Directory '{test_path}' does not exist.")
        sys.exit(1)

    for dir_path, _, files in os.walk(test_path):
        for file in files:
            if file.endswith(".cpp"):
                cpp_file_path = os.path.join(dir_path, file)
                compile_shared_object(cpp_file_path, dir_path)


if __name__ == "__main__":
    main()
