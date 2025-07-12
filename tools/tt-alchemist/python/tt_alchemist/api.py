# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""API for tt-alchemist library."""

import os
import ctypes
from pathlib import Path


class TTAlchemistAPI:
    """Singleton class for accessing tt-alchemist library functions."""

    _instance = None

    @classmethod
    def get_instance(cls):
        """Get the singleton instance of TTAlchemistAPI."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        """Initialize the API by loading the tt-alchemist library."""
        if TTAlchemistAPI._instance is not None:
            raise RuntimeError(
                "TTAlchemistAPI is a singleton. Use get_instance() instead."
            )

        self.lib = self._load_library()

        # Get the singleton instance from C++
        self.lib.tt_alchemist_TTAlchemist_getInstance.restype = ctypes.c_void_p
        self.instance_ptr = self.lib.tt_alchemist_TTAlchemist_getInstance()

        # Set up function argument types
        self.lib.tt_alchemist_TTAlchemist_modelToCpp.argtypes = [
            ctypes.c_void_p,  # instance pointer
            ctypes.c_char_p,  # input_file
        ]
        self.lib.tt_alchemist_TTAlchemist_modelToCpp.restype = ctypes.c_bool
        self.lib.tt_alchemist_TTAlchemist_generate.argtypes = [
            ctypes.c_void_p,  # instance pointer
            ctypes.c_char_p,  # input_file
            ctypes.c_char_p,  # output_dir
        ]
        self.lib.tt_alchemist_TTAlchemist_generate.restype = ctypes.c_bool

    def _load_library(self):
        """Load the tt-alchemist shared library.

        First tries to load from the package directory, then falls back to TT_MLIR_HOME.
        """
        # First try to load from the package directory
        package_dir = os.path.dirname(os.path.abspath(__file__))
        lib_path = os.path.join(package_dir, "lib", "libtt-alchemist-lib.so")

        if os.path.exists(lib_path):
            try:
                return ctypes.CDLL(lib_path)
            except Exception:
                # Fall back to TT_MLIR_HOME if loading from package fails
                pass

        # Fall back to TT_MLIR_HOME
        tt_mlir_home = os.environ.get("TT_MLIR_HOME")
        if not tt_mlir_home:
            raise RuntimeError(
                "Library not found in package and TT_MLIR_HOME environment variable is not set"
            )

        # Get the build directory name from environment or use default "build"
        build_dir = os.environ.get("BUILD_DIR", "build")

        # Load the tt-alchemist shared library
        lib_dir = os.path.join(tt_mlir_home, build_dir, "tools", "tt-alchemist", "csrc")

        if not lib_dir or not os.path.exists(lib_dir):
            raise RuntimeError(
                f"Could not find tt-alchemist library on path: {lib_dir}"
            )

        try:
            return ctypes.CDLL(f"{lib_dir}/libtt-alchemist-lib.so")
        except Exception as e:
            raise RuntimeError(f"Failed to load tt-alchemist library: {e}")

    def model_to_cpp(self, input_file):
        """Convert MLIR model to C++ code.

        Args:
            input_file: Path to the input MLIR model file.

        Returns:
            bool: True if conversion was successful, False otherwise.
        """
        if not isinstance(input_file, str):
            input_file = str(input_file)

        return self.lib.tt_alchemist_TTAlchemist_modelToCpp(
            self.instance_ptr, input_file.encode("utf-8")
        )

    def generate(self, input_file, output_dir):
        """Generate a standalone solution with the generated C++ code.

        This generates a directory with all necessary files to build and run the generated code,
        including CMakeLists.txt, precompiled headers, and a main C++ file.

        Args:
            input_file: Path to the input MLIR file.
            output_dir: Path to the output directory where the solution will be generated.

        Returns:
            bool: True if successful, False otherwise.
        """
        return self.lib.tt_alchemist_TTAlchemist_generate(
            self.instance_ptr,
            input_file.encode("utf-8"),
            output_dir.encode("utf-8"),
        )


# Convenience function for direct API usage
def model_to_cpp(input_file):
    """Convert MLIR model to C++ code.

    Args:
        input_file: Path to the input MLIR model file.

    Returns:
        bool: True if conversion was successful, False otherwise.
    """
    api = TTAlchemistAPI.get_instance()
    return api.model_to_cpp(input_file)


def generate(input_file, output_dir):
    """Generate a standalone solution with the generated C++ code.

    This generates a directory with all necessary files to build and run the generated code,
    including CMakeLists.txt, precompiled headers, and a main C++ file.

    Args:
        input_file: Path to the input MLIR file.
        output_dir: Path to the output directory where the solution will be generated.

    Returns:
        bool: True if successful, False otherwise.
    """
    api = TTAlchemistAPI.get_instance()
    return api.generate(input_file, output_dir)
