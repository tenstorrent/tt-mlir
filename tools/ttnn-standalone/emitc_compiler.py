#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
EmitC Compiler Module

This module provides functionality to compile EmitC-generated C++ files to shared
objects (.so) or standalone executables.

Usage:
    from emitc_compiler import EmitCCompiler, compile_emitc_to_so

    # Simple usage
    so_path = compile_emitc_to_so("/path/to/emitc.cpp")

    # Advanced usage with compiler instance
    compiler = EmitCCompiler(build_type="Release", mode="dylib")
    so_path = compiler.compile(cpp_file_path)
"""

import os
import subprocess
import shutil
from dataclasses import dataclass, field
from typing import Optional, List, Literal


class EmitCCompileError(Exception):
    """Raised when EmitC compilation fails."""

    pass


@dataclass
class EmitCCompiler:
    """
    Compiler for EmitC-generated C++ files.

    This class manages the compilation of EmitC C++ files to shared objects (.so)
    or standalone executables using CMake and Ninja.

    Attributes:
        build_type: CMake build type (Release, Debug, RelWithDebInfo, MinSizeRel)
        mode: Compilation mode - "dylib" for shared objects, "standalone" for executables
        incremental: If True, skip recompilation when output is newer than source
        metal_src_dir: Optional override for tt-metal source directory
        metal_lib_dir: Optional override for tt-metal library directory
        verbose: If True, print build progress messages

    Example:
        >>> compiler = EmitCCompiler(build_type="Release", mode="dylib")
        >>> so_path = compiler.compile("/path/to/emitc_output.cpp")
        >>> print(f"Compiled to: {so_path}")
    """

    build_type: str = "Release"
    mode: Literal["dylib", "standalone"] = "dylib"
    incremental: bool = False
    metal_src_dir: Optional[str] = None
    metal_lib_dir: Optional[str] = None
    verbose: bool = True

    # Internal state
    _cmake_configured: bool = field(default=False, init=False, repr=False)

    @staticmethod
    def get_standalone_dir() -> str:
        """Returns the ttnn-standalone directory path."""
        return os.path.dirname(os.path.abspath(__file__))

    @staticmethod
    def get_emitc_tests_path(build_dir: str) -> str:
        """Returns the path where EmitC TTNN tests live."""
        return os.path.join(build_dir, "test/ttmlir/EmitC/TTNN")

    @staticmethod
    def _is_file_older(f1_path: str, f2_path: str) -> bool:
        """Check if f1 is older than f2 based on modification time."""
        return os.path.getmtime(f1_path) < os.path.getmtime(f2_path)

    def _log(self, message: str) -> None:
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)

    def _run_cmake_setup(self) -> None:
        """
        Run CMake configuration for the ttnn-standalone project.

        This is called automatically before the first build and sets up
        the Ninja build system.

        Raises:
            EmitCCompileError: If CMake configuration fails
        """
        if self._cmake_configured:
            return

        standalone_source_dir = self.get_standalone_dir()
        standalone_build_dir = os.path.join(standalone_source_dir, "build")

        cmake_command = [
            "cmake",
            "-G",
            "Ninja",
            "-B",
            standalone_build_dir,
            "-S",
            standalone_source_dir,
            f"-DCMAKE_BUILD_TYPE={self.build_type}",
            f"-DCMAKE_C_COMPILER={os.environ.get('CC', 'clang')}",
            f"-DCMAKE_CXX_COMPILER={os.environ.get('CXX', 'clang++')}",
        ]

        if self.metal_src_dir:
            cmake_command.append(f"-DMETAL_SRC_DIR={self.metal_src_dir}")

        if self.metal_lib_dir:
            cmake_command.append(f"-DMETAL_LIB_DIR={self.metal_lib_dir}")

        try:
            subprocess.run(
                cmake_command,
                check=True,
                cwd=standalone_source_dir,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            raise EmitCCompileError(
                f"CMake configuration failed:\nstdout: {e.stdout}\nstderr: {e.stderr}"
            )

        self._cmake_configured = True

    def compile(
        self,
        cpp_file_path: str,
        output_dir: Optional[str] = None,
    ) -> str:
        """
        Compile a single C++ file to a shared object or executable.

        Args:
            cpp_file_path: Path to the .cpp file to compile
            output_dir: Directory for output file. If None, uses the same
                       directory as the input file.

        Returns:
            Path to the compiled output file (.so or .exe)

        Raises:
            EmitCCompileError: If compilation fails
            FileNotFoundError: If the input file doesn't exist
            ValueError: If the input file is not a .cpp file
        """
        # Validate input
        if not os.path.isfile(cpp_file_path):
            raise FileNotFoundError(f"Input file not found: {cpp_file_path}")

        if not cpp_file_path.endswith(".cpp"):
            raise ValueError(f"Input file must be a .cpp file: {cpp_file_path}")

        # Determine paths
        cpp_base_name = os.path.basename(cpp_file_path).rsplit(".", 1)[0]
        if output_dir is None:
            output_dir = os.path.dirname(cpp_file_path)

        standalone_source_dir = self.get_standalone_dir()
        standalone_build_dir = os.path.join(standalone_source_dir, "build")
        source_cpp_path = os.path.join(standalone_source_dir, f"ttnn-{self.mode}.cpp")
        ttnn_precompiled_header_path = os.path.join(
            standalone_source_dir, "ttnn-precompiled.hpp"
        )

        compiled_bin_path = os.path.join(
            standalone_build_dir,
            {"dylib": "libttnn-dylib.so", "standalone": "ttnn-standalone"}[self.mode],
        )

        extension = {"dylib": ".so", "standalone": ".exe"}[self.mode]
        output_file_name = cpp_base_name + extension
        destination_path = os.path.join(output_dir, output_file_name)

        # Check if rebuild is needed (incremental mode)
        if self.incremental and os.path.exists(destination_path):
            if self._is_file_older(
                cpp_file_path, destination_path
            ) and self._is_file_older(ttnn_precompiled_header_path, destination_path):
                self._log(
                    f"\nSkipping build for {cpp_base_name} - {output_file_name} is up to date"
                )
                return destination_path

        # Copy source file
        shutil.copy(cpp_file_path, source_cpp_path)

        # Configure CMake if needed
        self._run_cmake_setup()

        # Remove previous output if exists
        if os.path.exists(compiled_bin_path):
            os.remove(compiled_bin_path)

        # Build
        self._log(f"\nBuilding: {cpp_base_name}")
        build_command = [
            "cmake",
            "--build",
            standalone_build_dir,
            "--",
            f"ttnn-{self.mode}",
        ]

        try:
            subprocess.run(
                build_command,
                check=True,
                cwd=standalone_source_dir,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            raise EmitCCompileError(
                f"Build failed for {cpp_base_name}:\nstdout: {e.stdout}\nstderr: {e.stderr}"
            )

        self._log("  Build finished successfully!")

        # Verify output exists
        if not os.path.isfile(compiled_bin_path):
            raise EmitCCompileError(
                f"Build succeeded but output not found: {compiled_bin_path}"
            )

        # Copy to destination
        shutil.copy2(compiled_bin_path, destination_path)
        self._log(f"  Successfully copied compiled file to {destination_path}")

        return destination_path

    def compile_directory(
        self,
        directory: str,
        recursive: bool = True,
    ) -> List[str]:
        """
        Compile all .cpp files in a directory.

        Args:
            directory: Path to directory containing .cpp files
            recursive: If True, search subdirectories recursively

        Returns:
            List of paths to compiled output files

        Raises:
            EmitCCompileError: If any compilation fails
            NotADirectoryError: If the directory doesn't exist
        """
        if not os.path.isdir(directory):
            raise NotADirectoryError(f"Directory not found: {directory}")

        cpp_files = []
        if recursive:
            for dir_path, _, files in os.walk(directory):
                for file in files:
                    if file.endswith(".cpp"):
                        cpp_files.append(os.path.join(dir_path, file))
        else:
            for file in os.listdir(directory):
                if file.endswith(".cpp"):
                    cpp_files.append(os.path.join(directory, file))

        if not cpp_files:
            raise EmitCCompileError(f"No .cpp files found in: {directory}")

        output_paths = []
        for cpp_file in cpp_files:
            output_path = self.compile(cpp_file, output_dir=os.path.dirname(cpp_file))
            output_paths.append(output_path)

        return output_paths


def compile_emitc_to_so(
    cpp_file_path: str,
    output_dir: Optional[str] = None,
    build_type: str = "Release",
    incremental: bool = True,
    metal_src_dir: Optional[str] = None,
    metal_lib_dir: Optional[str] = None,
    verbose: bool = True,
) -> str:
    """
    Compile an EmitC .cpp file to a shared object (.so).

    This is a convenience function that creates an EmitCCompiler instance
    and compiles a single file.

    Args:
        cpp_file_path: Path to the .cpp file to compile
        output_dir: Directory for output .so file. If None, uses same directory as input.
        build_type: CMake build type (Release, Debug, etc.)
        incremental: If True, skip recompilation when .so is newer than .cpp
        verbose: If True, print progress messages

    Returns:
        Path to the compiled .so file

    Raises:
        EmitCCompileError: If compilation fails

    Example:
        >>> so_path = compile_emitc_to_so("/path/to/emitc.cpp")
        >>> print(f"Compiled: {so_path}")
    """
    compiler = EmitCCompiler(
        build_type=build_type,
        mode="dylib",
        incremental=incremental,
        metal_src_dir=metal_src_dir,
        metal_lib_dir=metal_lib_dir,
        verbose=verbose,
    )
    return compiler.compile(cpp_file_path, output_dir=output_dir)
