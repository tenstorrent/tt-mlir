#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Example: How to use tt-alchemist to generate TTNN unit tests from MLIR

This example demonstrates the complete workflow for generating parametrized
Python unit tests from TTNN MLIR operations.
"""

import os
import sys
import tempfile
import subprocess
from pathlib import Path

# Example TTNN MLIR with various operations
EXAMPLE_TTNN_MLIR = """
module attributes {ttnn.device = #ttnn.device<0>} {
  func.func @main(%arg0: tensor<1x32x128x128xbf16>,
                  %arg1: tensor<1x32x128x128xbf16>,
                  %arg2: tensor<1x64x256x256xbf16>,
                  %arg3: tensor<1x64x256x256xbf16>) -> tensor<1x32x128x128xbf16> {
    // First add operation with shape (1, 32, 128, 128)
    %0 = "ttnn.add"(%arg0, %arg1) : (tensor<1x32x128x128xbf16>, tensor<1x32x128x128xbf16>) -> tensor<1x32x128x128xbf16>

    // ReLU activation
    %1 = "ttnn.relu"(%0) : (tensor<1x32x128x128xbf16>) -> tensor<1x32x128x128xbf16>

    // Another add with different shape (1, 64, 256, 256)
    %2 = "ttnn.add"(%arg2, %arg3) : (tensor<1x64x256x256xbf16>, tensor<1x64x256x256xbf16>) -> tensor<1x64x256x256xbf16>

    // More operations for testing
    %3 = "ttnn.multiply"(%1, %1) : (tensor<1x32x128x128xbf16>, tensor<1x32x128x128xbf16>) -> tensor<1x32x128x128xbf16>
    %4 = "ttnn.exp"(%3) : (tensor<1x32x128x128xbf16>) -> tensor<1x32x128x128xbf16>
    %5 = "ttnn.sigmoid"(%4) : (tensor<1x32x128x128xbf16>) -> tensor<1x32x128x128xbf16>

    // Matrix multiplication
    %6 = "ttnn.matmul"(%arg0, %arg1) : (tensor<1x32x128x128xbf16>, tensor<1x32x128x128xbf16>) -> tensor<1x32x128x128xbf16>

    return %5 : tensor<1x32x128x128xbf16>
  }
}
"""

def step1_build_alchemist():
    """Step 1: Build tt-alchemist with the new test generation feature."""
    print("=" * 70)
    print("STEP 1: Building tt-alchemist")
    print("=" * 70)
    print()

    print("To build tt-alchemist with the test generation feature:")
    print()
    print("1. First, activate the environment:")
    print("   $ source env/activate")
    print()
    print("2. Configure the build:")
    print("   $ cmake -G Ninja -B build -DCMAKE_CXX_COMPILER_LAUNCHER=ccache")
    print()
    print("3. Build tt-alchemist:")
    print("   $ cmake --build build --target tt-alchemist")
    print()
    print("This will compile the C++ code and install the Python package.")
    print()

def step2_create_test_mlir():
    """Step 2: Create an example MLIR file."""
    print("=" * 70)
    print("STEP 2: Creating example MLIR file")
    print("=" * 70)
    print()

    # Create a temporary MLIR file
    mlir_file = "example_ttnn_ops.mlir"

    print(f"Creating {mlir_file} with various TTNN operations...")
    with open(mlir_file, 'w') as f:
        f.write(EXAMPLE_TTNN_MLIR)

    print(f"✓ Created {mlir_file}")
    print()
    print("The MLIR contains the following operations:")
    print("  - ttnn.add (2 instances with different shapes)")
    print("  - ttnn.relu")
    print("  - ttnn.multiply")
    print("  - ttnn.exp")
    print("  - ttnn.sigmoid")
    print("  - ttnn.matmul")
    print()

    return mlir_file

def step3_generate_tests_python():
    """Step 3: Generate tests using Python API."""
    print("=" * 70)
    print("STEP 3: Generating tests using Python API")
    print("=" * 70)
    print()

    print("Python code to generate tests:")
    print("-" * 40)

    code = '''
from tt_alchemist import generate_unit_tests

# Generate tests for all operations
generate_unit_tests(
    input_file="example_ttnn_ops.mlir",
    output_dir="generated_tests/",
    parametrized=True,      # Create parametrized tests
    verbose=True            # Show progress
)

# Generate tests for specific operations only
generate_unit_tests(
    input_file="example_ttnn_ops.mlir",
    output_dir="filtered_tests/",
    op_filter=["ttnn.add", "ttnn.relu"],  # Only these ops
    parametrized=True,
    verbose=True
)
'''

    print(code)
    print("-" * 40)
    print()

    # Try to import and run if available
    try:
        from tt_alchemist import generate_unit_tests

        print("Attempting to generate tests...")
        os.makedirs("generated_tests", exist_ok=True)

        result = generate_unit_tests(
            input_file="example_ttnn_ops.mlir",
            output_dir="generated_tests/",
            parametrized=True,
            verbose=True
        )

        if result:
            print("✓ Tests generated successfully!")
        else:
            print("✗ Test generation failed (may need full build)")
    except ImportError:
        print("Note: tt_alchemist not installed yet. Build it first using Step 1.")
    except Exception as e:
        print(f"Error: {e}")

    print()

def step4_expected_output():
    """Step 4: Show expected output structure."""
    print("=" * 70)
    print("STEP 4: Expected generated test structure")
    print("=" * 70)
    print()

    print("After successful generation, you should see:")
    print()
    print("generated_tests/")
    print("├── conftest.py              # Pytest fixtures for device setup")
    print("├── test_utils.py            # Helper functions for testing")
    print("├── test_ttnn_add.py         # Parametrized tests for add ops")
    print("├── test_ttnn_relu.py        # Tests for relu operations")
    print("├── test_ttnn_multiply.py    # Tests for multiply operations")
    print("├── test_ttnn_exp.py         # Tests for exp operations")
    print("├── test_ttnn_sigmoid.py     # Tests for sigmoid operations")
    print("└── test_ttnn_matmul.py      # Tests for matmul operations")
    print()

    print("Example of generated parametrized test (test_ttnn_add.py):")
    print("-" * 60)

    example_test = '''
import pytest
import torch
import ttnn
from test_utils import create_random_tensor, validate_output

class TestTtnnAdd:
    """Auto-generated tests for ttnn.add operation."""

    @pytest.mark.parametrize("shape", [
        (1, 32, 128, 128),    # From first add op
        (1, 64, 256, 256)     # From second add op
    ])
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    def test_add_parametrized(self, shape, dtype, device):
        """Test add with various parameters."""
        # Create random input tensors
        input0 = create_random_tensor(shape, dtype)
        input1 = create_random_tensor(shape, dtype)

        # Convert to TTNN tensors
        ttnn_input0 = ttnn.from_torch(input0, device=device)
        ttnn_input1 = ttnn.from_torch(input1, device=device)

        # Execute TTNN operation
        output = ttnn.add(ttnn_input0, ttnn_input1)

        # Compute golden values using PyTorch
        expected = torch.add(input0, input1)

        # Validate output
        actual = ttnn.to_torch(output)
        assert validate_output(actual, expected)
'''

    print(example_test)
    print("-" * 60)
    print()

def step5_run_tests():
    """Step 5: Show how to run the generated tests."""
    print("=" * 70)
    print("STEP 5: Running the generated tests")
    print("=" * 70)
    print()

    print("Once tests are generated, you can run them with pytest:")
    print()
    print("1. Run all generated tests:")
    print("   $ pytest generated_tests/")
    print()
    print("2. Run tests for specific operation:")
    print("   $ pytest generated_tests/test_ttnn_add.py")
    print()
    print("3. Run with verbose output:")
    print("   $ pytest -v generated_tests/")
    print()
    print("4. Run specific test case:")
    print("   $ pytest generated_tests/test_ttnn_add.py::TestTtnnAdd::test_add_parametrized")
    print()
    print("5. Run with specific parameters:")
    print("   $ pytest generated_tests/ -k 'shape0-1x32x128x128'")
    print()

def step6_advanced_usage():
    """Step 6: Show advanced usage examples."""
    print("=" * 70)
    print("STEP 6: Advanced usage")
    print("=" * 70)
    print()

    print("Advanced examples:")
    print()

    print("1. Generate non-parametrized individual tests:")
    print("-" * 40)
    code1 = '''
generate_unit_tests(
    input_file="model.mlir",
    output_dir="individual_tests/",
    parametrized=False,  # Each op gets its own test method
    verbose=True
)
'''
    print(code1)
    print()

    print("2. Generate from MLIR string instead of file:")
    print("-" * 40)
    code2 = '''
from tt_alchemist import TTAlchemistAPI

api = TTAlchemistAPI.get_instance()

mlir_string = """
module {
    func.func @main(%arg0: tensor<1x32x32x32xbf16>) -> tensor<1x32x32x32xbf16> {
        %0 = "ttnn.relu"(%arg0) : (tensor<1x32x32x32xbf16>) -> tensor<1x32x32x32xbf16>
        return %0 : tensor<1x32x32x32xbf16>
    }
}
"""

api.generate_unit_tests_from_string(
    mlir_string=mlir_string,
    output_dir="string_tests/",
    parametrized=True
)
'''
    print(code2)
    print()

    print("3. Filter multiple operations:")
    print("-" * 40)
    code3 = '''
generate_unit_tests(
    input_file="large_model.mlir",
    output_dir="filtered_tests/",
    op_filter=["ttnn.add", "ttnn.multiply", "ttnn.matmul"],
    parametrized=True,
    verbose=True
)
'''
    print(code3)
    print()

def main():
    """Run all example steps."""
    print()
    print("🚀 TT-ALCHEMIST UNIT TEST GENERATION EXAMPLE")
    print("=" * 70)
    print()
    print("This example demonstrates how to use the new unit test generation")
    print("feature in tt-alchemist to automatically create Python tests from")
    print("TTNN MLIR operations.")
    print()

    # Run all steps
    step1_build_alchemist()
    mlir_file = step2_create_test_mlir()
    step3_generate_tests_python()
    step4_expected_output()
    step5_run_tests()
    step6_advanced_usage()

    print("=" * 70)
    print("📚 SUMMARY")
    print("=" * 70)
    print()
    print("The unit test generation feature allows you to:")
    print("  ✓ Automatically generate pytest tests from TTNN MLIR")
    print("  ✓ Create parametrized tests for similar operations")
    print("  ✓ Filter which operations to test")
    print("  ✓ Generate complete test infrastructure (fixtures, utilities)")
    print()
    print("Key benefits:")
    print("  • Reduces manual test writing effort")
    print("  • Ensures comprehensive test coverage")
    print("  • Maintains consistency across tests")
    print("  • Supports easy test maintenance")
    print()
    print("For more information, see:")
    print("  tools/tt-alchemist/README_UNIT_TEST_GEN.md")
    print()

if __name__ == "__main__":
    main()