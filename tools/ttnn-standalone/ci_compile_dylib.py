#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


import sys
import os
import subprocess
import shutil
import argparse


# Returns the path where EmitC TTNN tests live
#
def get_emitc_tests_path(build_dir):
    if get_emitc_tests_path.path:
        return get_emitc_tests_path.path

    get_emitc_tests_path.path = os.path.join(build_dir, "test/ttmlir/EmitC/TTNN")

    return get_emitc_tests_path.path


# Returns ttnn-standalone dir
#
def get_standalone_dir():
    # Calculate this relative to this script's location
    #
    return os.path.dirname(os.path.abspath(__file__))


# Runs cmake setup for .so compilation
# Runs only once per script
#
def run_cmake_setup(args):
    if run_cmake_setup.already_created:
        return

    standalone_source_dir = get_standalone_dir()
    standalone_build_dir = os.path.join(standalone_source_dir, "build")
    cmake_command = [
        "cmake",
        "-G",
        "Ninja",
        "-B",
        standalone_build_dir,
        "-S",
        standalone_source_dir,
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CXX_COMPILER=clang++",
    ]

    if args.metal_src_dir:
        cmake_command.append(f"-DMETAL_SRC_DIR={args.metal_src_dir}")

    if args.metal_lib_dir:
        cmake_command.append(f"-DMETAL_LIB_DIR={args.metal_lib_dir}")

    # Print metal_src_dir and metal_lib_dir
    print(f"Setting up cmake environment with:")
    print(f"  METAL_SRC_DIR: {args.metal_src_dir}")
    print(f"  METAL_LIB_DIR: {args.metal_lib_dir}")

    try:
        result = subprocess.run(
            cmake_command,
            check=True,
            cwd=standalone_source_dir,
            # capture_output=True,
            text=True,
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
def compile_shared_object(cpp_file_path, output_dir, args):
    # Base name of the provided cpp file
    #
    cpp_base_name = os.path.basename(cpp_file_path).rsplit(".", 1)[0]

    # Various dirs
    #
    standalone_source_dir = get_standalone_dir()
    standalone_build_dir = os.path.join(standalone_source_dir, "build")
    source_cpp_path = os.path.join(standalone_source_dir, "ttnn-dylib.cpp")
    compiled_so_path = os.path.join(standalone_build_dir, "libttnn-dylib.so")

    try:
        # Copy provided cpp file to source dir
        #
        shutil.copy(cpp_file_path, source_cpp_path)

        # Run cmake setup command first
        #
        run_cmake_setup(args)

        # Remove previous .so if exists
        if os.path.exists(compiled_so_path):
            os.remove(compiled_so_path)

        # Run build
        #
        print(f"\nBuilding: {cpp_base_name}")
        build_command = [
            "cmake",
            "--build",
            standalone_build_dir,
            "--verbose",
            "--",
            "ttnn-dylib",
        ]
        result = subprocess.run(
            build_command,
            check=True,
            cwd=standalone_source_dir,
            capture_output=True,
            text=True,
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


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compile EmitC TTNN tests to shared objects."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--build-dir",
        dest="build_dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Specify a custom build directory instead of the default 'build' directory",
    )
    group.add_argument(
        "--file",
        dest="file",
        type=str,
        default=None,
        metavar="FILE",
        help="Specify a single cpp file for compilation",
    )

    # Add option to override metal-src-dir and metal-lib-dir
    #
    # TODO: Verify that either all group arguments are provided or none
    #
    group = parser.add_argument_group(
        "dirs", description="Directories needed from copilation"
    )
    parser.add_argument(
        "--metal-src-dir",
        type=str,
    )
    parser.add_argument(
        "--metal-lib-dir",
        type=str,
    )
    return parser.parse_args()


def main():
    print("PRINTING ENVS CCD")
    print("I'm in ci_compile_dylib.py")

    # Unrolled version of the loop
    var_name = "TT_METAL_HOME"
    var_value = os.environ.get(var_name)
    print(f"  {var_name} environment variable: {var_value if var_value else 'not set'}")

    var_name = "CMAKE_INSTALL_PREFIX"
    var_value = os.environ.get(var_name)
    print(f"  {var_name} environment variable: {var_value if var_value else 'not set'}")

    var_name = "TT_MLIR_HOME"
    var_value = os.environ.get(var_name)
    print(f"  {var_name} environment variable: {var_value if var_value else 'not set'}")

    var_name = "FORGE_HOME"
    var_value = os.environ.get(var_name)
    print(f"  {var_name} environment variable: {var_value if var_value else 'not set'}")

    print(f"  PRINTING FROM: {__file__}")

    args = parse_arguments()
    build_dir = args.build_dir
    file = args.file

    # Enumerate files for compilation
    cpp_files = []

    if file:
        print(f"Using custom file for compilation: {file}")

        if not os.path.isfile(file) or not file.endswith(".cpp"):
            print(f"Error: File '{file}' does not exist or is not a .cpp file.")
            sys.exit(1)

        cpp_files.append(file)
    else:
        if not build_dir:
            build_dir = os.path.join(
                get_standalone_dir(),
                "../../build",
            )

        test_path = get_emitc_tests_path(build_dir)

        if not os.path.isdir(test_path):
            print(f"Error: Test path directory '{test_path}' does not exist.")
            sys.exit(1)

        print(f"Using test path for compilation: {test_path}")

        for dir_path, _, files in os.walk(test_path):
            for file in files:
                if file.endswith(".cpp"):
                    cpp_file_path = os.path.join(dir_path, file)
                    cpp_files.append(cpp_file_path)

    for cpp_file in cpp_files:
        compile_shared_object(
            cpp_file_path=cpp_file,
            output_dir=os.path.dirname(cpp_file),
            args=args,
        )

    print("Compilation completed successfully!")


if __name__ == "__main__":
    main()
