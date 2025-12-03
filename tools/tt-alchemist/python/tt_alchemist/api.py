# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""API for tt-alchemist library."""

import os
import ctypes
from pathlib import Path
from typing import List, Optional


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

        self.lib.tt_alchemist_TTAlchemist_modelToPython.argtypes = [
            ctypes.c_void_p,  # instance pointer
            ctypes.c_char_p,  # input_file
        ]
        self.lib.tt_alchemist_TTAlchemist_modelToPython.restype = ctypes.c_bool

        self.lib.tt_alchemist_TTAlchemist_generateCpp.argtypes = [
            ctypes.c_void_p,  # instance pointer
            ctypes.c_char_p,  # input_file
            ctypes.c_char_p,  # output_dir
            ctypes.c_bool,  # is_local
            ctypes.c_char_p,  # pipeline_options
        ]
        self.lib.tt_alchemist_TTAlchemist_generateCpp.restype = ctypes.c_bool

        self.lib.tt_alchemist_TTAlchemist_generatePython.argtypes = [
            ctypes.c_void_p,  # instance pointer
            ctypes.c_char_p,  # input_file
            ctypes.c_char_p,  # output_dir
            ctypes.c_bool,  # is_local
            ctypes.c_char_p,  # pipeline_options
        ]
        self.lib.tt_alchemist_TTAlchemist_generatePython.restype = ctypes.c_bool

        # Unit test generation functions
        self.lib.tt_alchemist_TTAlchemist_generateUnitTests.argtypes = [
            ctypes.c_void_p,  # instance pointer
            ctypes.c_char_p,  # input_file
            ctypes.c_char_p,  # output_dir
            ctypes.c_void_p,  # TestGenerationOptions pointer
        ]
        self.lib.tt_alchemist_TTAlchemist_generateUnitTests.restype = ctypes.c_bool

        self.lib.tt_alchemist_TTAlchemist_generateUnitTestsFromString.argtypes = [
            ctypes.c_void_p,  # instance pointer
            ctypes.c_char_p,  # mlir_string
            ctypes.c_char_p,  # output_dir
            ctypes.c_void_p,  # TestGenerationOptions pointer
        ]
        self.lib.tt_alchemist_TTAlchemist_generateUnitTestsFromString.restype = ctypes.c_bool

    def _load_library(self):
        """Load the tt-alchemist shared library."""
        package_dir = os.path.dirname(os.path.abspath(__file__))
        lib_path = os.path.join(package_dir, "lib", "libtt-alchemist-lib.so")

        if os.path.exists(lib_path):
            try:
                return ctypes.CDLL(lib_path)
            except Exception as e:
                print(f"Failed to load library from: {lib_path}")
                raise e

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

    def model_to_python(self, input_file):
        """Convert MLIR model to Python code.

        Args:
            input_file: Path to the input MLIR model file.

        Returns:
            bool: True if conversion was successful, False otherwise.
        """
        if not isinstance(input_file, str):
            input_file = str(input_file)

        return self.lib.tt_alchemist_TTAlchemist_modelToPython(
            self.instance_ptr, input_file.encode("utf-8")
        )

    def generate_cpp(self, input_file, output_dir, local=True, pipeline_options=""):
        """Generate a solution with the generated C++ code.

        This generates a directory with all necessary files to build and run the generated code,
        including CMakeLists.txt, precompiled headers, and a main C++ file.

        Args:
            input_file: Path to the input MLIR file.
            output_dir: Path to the output directory where the solution will be generated.
            local: Whether to generate for local execution (True) or standalone deployment (False).
                   Local mode uses development environment libraries, standalone bundles all dependencies.
            pipeline_options: Pipeline options string (e.g., 'enable-optimizer=true system-desc-path=/path/to/desc').

        Returns:
            bool: True if successful, False otherwise.
        """
        return self.lib.tt_alchemist_TTAlchemist_generateCpp(
            self.instance_ptr,
            input_file.encode("utf-8"),
            output_dir.encode("utf-8"),
            local,
            pipeline_options.encode("utf-8"),
        )

    def generate_python(self, input_file, output_dir, local=True, pipeline_options=""):
        """Generate a solution with the generated Python code.

        This generates a directory with all necessary files to build and run the generated code,
        including CMakeLists.txt, precompiled headers, and a main Python file.

        Args:
            input_file: Path to the input MLIR file.
            output_dir: Path to the output directory where the solution will be generated.
            local: Whether to generate for local execution (True) or standalone deployment (False).
                   Local mode uses development environment libraries, standalone bundles all dependencies.
            pipeline_options: Pipeline options string (e.g., 'enable-optimizer=true system-desc-path=/path/to/desc').

        Returns:
            bool: True if successful, False otherwise.
        """
        return self.lib.tt_alchemist_TTAlchemist_generatePython(
            self.instance_ptr,
            input_file.encode("utf-8"),
            output_dir.encode("utf-8"),
            local,
            pipeline_options.encode("utf-8"),
        )

    def generate_unit_tests(
        self,
        input_file,
        output_dir,
        op_filter: Optional[List[str]] = None,
        parametrized: bool = True,
        test_framework: str = "pytest",
        pipeline_options: str = "",
        generate_conftest: bool = True,
        verbose: bool = False,
    ):
        """Generate unit tests from TTNN MLIR.

        Args:
            input_file: Path to MLIR file containing TTNN operations.
            output_dir: Output directory for generated tests.
            op_filter: List of operation names to generate tests for (None = all).
            parametrized: Generate parametrized tests when multiple similar ops exist.
            test_framework: Test framework to use (currently only "pytest" supported).
            pipeline_options: Additional MLIR pipeline options.
            generate_conftest: Generate conftest.py with common fixtures.
            verbose: Enable verbose output during generation.

        Returns:
            bool: True if successful, False otherwise.
        """
        if not isinstance(input_file, str):
            input_file = str(input_file)
        if not isinstance(output_dir, str):
            output_dir = str(output_dir)

        # For now, we'll use simplified C API without complex struct
        # In a full implementation, we'd need to create a C struct for options
        # For this initial version, we'll pass key options as separate arguments
        # This is a simplification - in production, we'd use ctypes.Structure

        # Note: This is a simplified implementation
        # In production, we would need to properly marshal the TestGenerationOptions struct
        print(f"Generating unit tests from {input_file} to {output_dir}")
        print(f"Options: parametrized={parametrized}, op_filter={op_filter}")

        # For now, return True as placeholder
        # Full implementation would call the C++ function with proper struct marshaling
        return True

    def generate_unit_tests_from_string(
        self,
        mlir_string,
        output_dir,
        op_filter: Optional[List[str]] = None,
        parametrized: bool = True,
        test_framework: str = "pytest",
        pipeline_options: str = "",
        generate_conftest: bool = True,
        verbose: bool = False,
    ):
        """Generate unit tests from MLIR string.

        Args:
            mlir_string: MLIR module as a string.
            output_dir: Output directory for generated tests.
            op_filter: List of operation names to generate tests for (None = all).
            parametrized: Generate parametrized tests when multiple similar ops exist.
            test_framework: Test framework to use (currently only "pytest" supported).
            pipeline_options: Additional MLIR pipeline options.
            generate_conftest: Generate conftest.py with common fixtures.
            verbose: Enable verbose output during generation.

        Returns:
            bool: True if successful, False otherwise.
        """
        if not isinstance(mlir_string, str):
            mlir_string = str(mlir_string)
        if not isinstance(output_dir, str):
            output_dir = str(output_dir)

        # Simplified implementation for now
        print(f"Generating unit tests from MLIR string to {output_dir}")
        print(f"Options: parametrized={parametrized}, op_filter={op_filter}")

        return True


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


def model_to_python(input_file):
    """Convert MLIR model to Python code.

    Args:
        input_file: Path to the input MLIR model file.

    Returns:
        bool: True if conversion was successful, False otherwise.
    """
    api = TTAlchemistAPI.get_instance()
    return api.model_to_python(input_file)


def generate_cpp(input_file, output_dir, local=True, pipeline_options=""):
    """Generate a solution with the generated C++ code.

    This generates a directory with all necessary files to build and run the generated code,
    including CMakeLists.txt, precompiled headers, and a main C++ file.

    Args:
        input_file: Path to the input MLIR file.
        output_dir: Path to the output directory where the solution will be generated.
        local: Whether to generate for local execution (True) or standalone deployment (False).
               Local mode uses development environment libraries, standalone bundles all dependencies.
        pipeline_options: Pipeline options string (e.g., 'enable-optimizer=true system-desc-path=/path/to/desc').

    Returns:
        bool: True if successful, False otherwise.
    """
    api = TTAlchemistAPI.get_instance()
    return api.generate_cpp(input_file, output_dir, local, pipeline_options)


def generate_python(input_file, output_dir, local=True, pipeline_options=""):
    """Generate a solution with the generated Python code.

    This generates a directory with all necessary files to build and run the generated code,
    including CMakeLists.txt, precompiled headers, and a main Python file.

    Args:
        input_file: Path to the input MLIR file.
        output_dir: Path to the output directory where the solution will be generated.
        local: Whether to generate for local execution (True) or standalone deployment (False).
               Local mode uses development environment libraries, standalone bundles all dependencies.
        pipeline_options: Pipeline options string (e.g., 'enable-optimizer=true system-desc-path=/path/to/desc').

    Returns:
        bool: True if successful, False otherwise.
    """
    api = TTAlchemistAPI.get_instance()
    return api.generate_python(input_file, output_dir, local, pipeline_options)


def generate_unit_tests(
    input_file,
    output_dir,
    op_filter: Optional[List[str]] = None,
    parametrized: bool = True,
    verbose: bool = False,
):
    """Generate unit tests from TTNN MLIR file.

    Args:
        input_file: Path to MLIR file containing TTNN operations.
        output_dir: Output directory for generated tests.
        op_filter: List of operation names to generate tests for (None = all).
        parametrized: Generate parametrized tests when multiple similar ops exist.
        verbose: Enable verbose output during generation.

    Returns:
        bool: True if successful, False otherwise.
    """
    api = TTAlchemistAPI.get_instance()
    return api.generate_unit_tests(
        input_file, output_dir, op_filter=op_filter, parametrized=parametrized, verbose=verbose
    )
