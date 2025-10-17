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
    # Calculate standalone dir path by using this script's location
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
        f"-DCMAKE_BUILD_TYPE={args.build_type}",
        f"-DCMAKE_C_COMPILER={os.environ.get('CC', 'clang')}",
        f"-DCMAKE_CXX_COMPILER={os.environ.get('CXX', 'clang++')}",
    ]

    if args.metal_src_dir:
        cmake_command.append(f"-DMETAL_SRC_DIR={args.metal_src_dir}")

    if args.metal_lib_dir:
        cmake_command.append(f"-DMETAL_LIB_DIR={args.metal_lib_dir}")

    try:
        result = subprocess.run(
            cmake_command,
            check=True,
            cwd=standalone_source_dir,
            capture_output=True,
            text=True,
        )
    except Exception as e:
        print(f"Error setting up cmake environment: {e}")
        print(e.stderr)
        print(e.stdout)
        sys.exit(1)

    run_cmake_setup.already_created = True


# Init fn variables
#
get_emitc_tests_path.path = None
run_cmake_setup.already_created = None


# Check if the modification time of file1 is lesser than file2 (i.e. if file1 is older than file2)
#
def is_file_older(f1_path, f2_path):
    f1_mtime = os.path.getmtime(f1_path)
    f2_mtime = os.path.getmtime(f2_path)
    return f1_mtime < f2_mtime


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
    source_cpp_path = os.path.join(standalone_source_dir, f"ttnn-{args.mode}.cpp")
    ttnn_precompiled_header_path = os.path.join(
        standalone_source_dir, "ttnn-precompiled.hpp"
    )
    compiled_bin_path = os.path.join(
        standalone_build_dir,
        {
            "dylib": "libttnn-dylib.so",
            "standalone": "ttnn-standalone",
        }[args.mode],
    )

    extension = {
        "dylib": ".so",
        "standalone": ".exe",
    }[args.mode]

    # Determine output .so path
    output_file_name = cpp_base_name + extension
    destination_path = os.path.join(output_dir, output_file_name)

    # If the build is run in incremental mode, check if rebuild is needed by comparing modification times
    if args.incremental and os.path.exists(destination_path):
        if is_file_older(cpp_file_path, destination_path) and is_file_older(
            ttnn_precompiled_header_path, destination_path
        ):
            print(
                f"\nSkipping build for {cpp_base_name} - {output_file_name} file is up to date"
            )
            return

    try:
        # Copy provided cpp file to source dir
        #
        shutil.copy(cpp_file_path, source_cpp_path)

        # Run cmake setup command first
        #
        run_cmake_setup(args)

        # Remove previous .so if exists
        if os.path.exists(compiled_bin_path):
            os.remove(compiled_bin_path)

        # Run build
        #
        print(f"\nBuilding: {cpp_base_name}")
        build_command = [
            "cmake",
            "--build",
            standalone_build_dir,
            "--",
            f"ttnn-{args.mode}",
        ]
        subprocess.run(
            build_command,
            check=True,
            cwd=standalone_source_dir,
            capture_output=True,
            text=True,
        )
        print(f"  Build finished successfully!")

        # Confirm .so exists
        if not os.path.isfile(compiled_bin_path):
            print(f"Error: Compiled file '{compiled_bin_path}' not found.")
            sys.exit(1)

        # Copy the compiled .so
        #
        shutil.copy2(compiled_bin_path, destination_path)
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
        "--dir",
        dest="dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Specify a directory filled with cpp sources for compilation",
    )
    group.add_argument(
        "--file",
        dest="file",
        type=str,
        default=None,
        metavar="FILE",
        help="Specify a single cpp file for compilation",
    )
    parser.add_argument(
        "--build-type",
        type=str,
        default="Release",
        help="Specify a custom build type for CMAKE_BUILD_TYPE",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="dylib",
        choices=["dylib", "standalone"],
        help="Compile dylib or standalone binaries",
    )
    parser.add_argument(
        "-i",
        "--incremental",
        dest="incremental",
        action="store_true",
        help="Incremental build mode. Only rebuilds files that have changed since last build. "
        "NOTE: Build flag changes won't trigger rebuilds.",
    )

    # Add option to override metal-src-dir and metal-lib-dir
    #
    group = parser.add_argument_group(
        "dirs", description="Overrides for metal-src-dir and metal-lib-dir"
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
    args = parse_arguments()
    build_dir = args.build_dir if args.build_dir is not None else args.dir
    needs_test_path = args.dir is None
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
        # If not even build_dir is provided, use default test path in tt-mlir
        if not build_dir:
            if not "TT_MLIR_HOME" in os.environ:
                print(
                    "Tried building tests from default path - TT_MLIR_HOME env not found, exiting!"
                )
                sys.exit(1)
            build_dir = os.path.join(os.environ["TT_MLIR_HOME"], "build")
            print(f"Building tt-mlir tests in {build_dir}")

        test_path = get_emitc_tests_path(build_dir) if needs_test_path else build_dir

        if not os.path.isdir(test_path):
            print(f"Error: Test path directory '{test_path}' does not exist.")
            sys.exit(1)

        print(f"Using test path for compilation: {test_path}")

        for dir_path, _, files in os.walk(test_path):
            for file in files:
                if file.endswith(".cpp"):
                    cpp_file_path = os.path.join(dir_path, file)
                    cpp_files.append(cpp_file_path)

    if not cpp_files:
        print("Error: No .cpp files found for compilation.")
        sys.exit(1)

    for cpp_file in cpp_files:
        compile_shared_object(
            cpp_file_path=cpp_file,
            output_dir=os.path.dirname(cpp_file),
            args=args,
        )

    print("Compilation completed successfully!")


if __name__ == "__main__":
    main()
