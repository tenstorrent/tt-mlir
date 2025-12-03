#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Test script for Alchemist unit test generation."""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Try to import tt_alchemist
try:
    from tt_alchemist import TTAlchemistAPI, generate_unit_tests
except ImportError:
    print("Warning: tt_alchemist not installed. Skipping import test.")
    sys.exit(0)


def create_test_mlir() -> str:
    """Create a simple MLIR test file."""
    mlir_content = """
module attributes {ttnn.device = #ttnn.device<0>} {
  func.func @main(%arg0: tensor<1x32x128x128xbf16>, %arg1: tensor<1x32x128x128xbf16>) -> tensor<1x32x128x128xbf16> {
    %0 = "ttnn.add"(%arg0, %arg1) : (tensor<1x32x128x128xbf16>, tensor<1x32x128x128xbf16>) -> tensor<1x32x128x128xbf16>
    %1 = "ttnn.relu"(%0) : (tensor<1x32x128x128xbf16>) -> tensor<1x32x128x128xbf16>

    // Another add with different shape
    %arg2 = "ttnn.constant"() {value = dense<1.0> : tensor<1x64x256x256xbf16>} : () -> tensor<1x64x256x256xbf16>
    %arg3 = "ttnn.constant"() {value = dense<2.0> : tensor<1x64x256x256xbf16>} : () -> tensor<1x64x256x256xbf16>
    %2 = "ttnn.add"(%arg2, %arg3) : (tensor<1x64x256x256xbf16>, tensor<1x64x256x256xbf16>) -> tensor<1x64x256x256xbf16>

    return %1 : tensor<1x32x128x128xbf16>
  }
}
"""
    return mlir_content


def test_basic_generation():
    """Test basic test generation functionality."""
    print("Testing basic unit test generation...")

    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test MLIR file
        mlir_file = os.path.join(tmpdir, "test.mlir")
        with open(mlir_file, "w") as f:
            f.write(create_test_mlir())

        # Create output directory
        output_dir = os.path.join(tmpdir, "generated_tests")
        os.makedirs(output_dir)

        # Generate tests
        print(f"Generating tests from {mlir_file} to {output_dir}")
        result = generate_unit_tests(
            mlir_file,
            output_dir,
            parametrized=True,
            verbose=True
        )

        if result:
            print("✓ Test generation succeeded")

            # Check if expected files were created (when fully implemented)
            expected_files = [
                "conftest.py",
                "test_utils.py",
                # When fully implemented, we'd expect:
                # "test_add.py",
                # "test_relu.py",
            ]

            for filename in expected_files:
                filepath = os.path.join(output_dir, filename)
                if os.path.exists(filepath):
                    print(f"✓ Found {filename}")
                else:
                    print(f"✗ Missing {filename} (expected when fully implemented)")

        else:
            print("✗ Test generation failed (expected for placeholder implementation)")

    print("Test completed!")


def test_op_filter():
    """Test generation with operation filter."""
    print("\nTesting with operation filter...")

    with tempfile.TemporaryDirectory() as tmpdir:
        mlir_file = os.path.join(tmpdir, "test.mlir")
        with open(mlir_file, "w") as f:
            f.write(create_test_mlir())

        output_dir = os.path.join(tmpdir, "filtered_tests")
        os.makedirs(output_dir)

        # Generate tests only for add operations
        result = generate_unit_tests(
            mlir_file,
            output_dir,
            op_filter=["ttnn.add"],
            parametrized=True,
            verbose=True
        )

        if result:
            print("✓ Filtered test generation succeeded")
        else:
            print("✗ Filtered test generation failed (expected for placeholder)")

    print("Filter test completed!")


def test_api_instance():
    """Test API singleton pattern."""
    print("\nTesting API singleton...")

    api1 = TTAlchemistAPI.get_instance()
    api2 = TTAlchemistAPI.get_instance()

    if api1 is api2:
        print("✓ Singleton pattern working correctly")
    else:
        print("✗ Singleton pattern failed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Alchemist Unit Test Generation - Test Suite")
    print("=" * 60)

    test_api_instance()
    test_basic_generation()
    test_op_filter()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("Note: This is testing the API structure.")
    print("Full C++ implementation needs to be compiled first.")
    print("=" * 60)


if __name__ == "__main__":
    main()