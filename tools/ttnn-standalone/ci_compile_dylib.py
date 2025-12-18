#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
CLI wrapper for EmitC compiler.

This script compiles EmitC-generated C++ files to shared objects (.so) or
standalone executables.

Usage:
    python ci_compile_dylib.py --file path/to/file.cpp
    python ci_compile_dylib.py --dir path/to/cpp/files
    python ci_compile_dylib.py --build-dir path/to/build
"""

import sys
import os
import argparse

from emitc_compiler import EmitCCompiler, EmitCCompileError


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

    # Create compiler instance
    compiler = EmitCCompiler(
        build_type=args.build_type,
        mode=args.mode,
        incremental=args.incremental,
        metal_src_dir=args.metal_src_dir,
        metal_lib_dir=args.metal_lib_dir,
        verbose=True,
    )

    try:
        if args.file:
            # Single file mode
            print(f"Using custom file for compilation: {args.file}")
            compiler.compile(args.file)

        else:
            # Directory mode
            build_dir = args.build_dir if args.build_dir is not None else args.dir
            needs_test_path = args.dir is None

            if not build_dir:
                if "TT_MLIR_HOME" not in os.environ:
                    print(
                        "Tried building tests from default path - TT_MLIR_HOME env not found, exiting!"
                    )
                    sys.exit(1)
                build_dir = os.path.join(os.environ["TT_MLIR_HOME"], "build")
                print(f"Building tt-mlir tests in {build_dir}")

            test_path = (
                EmitCCompiler.get_emitc_tests_path(build_dir)
                if needs_test_path
                else build_dir
            )

            print(f"Using test path for compilation: {test_path}")
            compiler.compile_directory(test_path)

        print("Compilation completed successfully!")

    except (EmitCCompileError, FileNotFoundError, NotADirectoryError) as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
