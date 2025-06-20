# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Python bindings for the tt-alchemist C++ library
"""

from enum import Enum

# These enums mirror the C++ enums
class OptimizationLevel(Enum):
    MINIMAL = 0
    NORMAL = 1
    AGGRESSIVE = 2


class BuildFlavor(Enum):
    RELEASE = 0
    DEBUG = 1
    PROFILE = 2


class HardwareTarget(Enum):
    GRAYSKULL = 0
    WORMHOLE = 1
    BLACKHOLE = 2


# This is a placeholder for the actual C++ bindings
# In a real implementation, this would be generated using pybind11
class TTAlchemist:
    def __init__(self):
        self.last_error = ""

    def model_to_cpp(self, input_file, config):
        """
        Convert a model to C++

        Args:
            input_file: Path to the input MLIR file
            config: Configuration for the conversion

        Returns:
            bool: True if successful, False otherwise
        """
        # This is a placeholder implementation
        # In a real implementation, this would call the C++ library
        print(
            f"Converting {input_file} to C++ with {config['opt_level']} optimization..."
        )
        return True

    def build_solution(self, model_dir, config):
        """
        Build a generated solution

        Args:
            model_dir: Path to the model directory
            config: Configuration for the build

        Returns:
            bool: True if successful, False otherwise
        """
        # This is a placeholder implementation
        # In a real implementation, this would call the C++ library
        print(
            f"Building {model_dir} with {config['flavor']} flavor for {config['target']}..."
        )
        return True

    def run_solution(self, model_dir, config):
        """
        Run a built solution

        Args:
            model_dir: Path to the model directory
            config: Configuration for the run

        Returns:
            bool: True if successful, False otherwise
        """
        # This is a placeholder implementation
        # In a real implementation, this would call the C++ library
        print(f"Running {model_dir}...")
        return True

    def profile_solution(self, model_dir, config, report_file):
        """
        Profile a built solution

        Args:
            model_dir: Path to the model directory
            config: Configuration for the run
            report_file: Path to the output report file

        Returns:
            bool: True if successful, False otherwise
        """
        # This is a placeholder implementation
        # In a real implementation, this would call the C++ library
        print(f"Profiling {model_dir}...")
        return True

    def get_last_error(self):
        """
        Get the last error message

        Returns:
            str: The last error message
        """
        return self.last_error
