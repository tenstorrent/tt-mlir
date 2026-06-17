# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Example: How to add a new pattern with integrated tests.

This is a template/example showing how to structure a pattern file
with integrated test metadata. Copy this file and modify it for your
own patterns.
"""

import torch
import d2m_jit as d2m
from ttmlir.dialects import ttir


# ============================================================================
# Step 1: Define the kernel function(s)
# ============================================================================


@d2m.kernel
def my_fused_kernel(in1, in2, out, m_blocks, n_blocks):
    """Example kernel that performs some operation.

    Replace this with your actual kernel implementation.
    """
    m_off = core_index(0) * m_blocks
    n_off = core_index(1) * n_blocks
    for m in range(m_blocks):
        for n in range(n_blocks):
            x1 = remote_load(in1, [m_off + m, n_off + n])
            x2 = remote_load(in2, [m_off + m, n_off + n])
            # Example: element-wise multiply
            result = x1 * x2
            remote_store(out, [m_off + m, n_off + n], result)


# ============================================================================
# Step 2: Define the pattern rewrite rule(s)
# ============================================================================


@d2m.pattern(root=ttir.MulOp, benefit=10)
def lower_my_pattern(op, rewriter):
    """Replace ttir.mul with the fused d2m subgraph.

    This is where you define how to recognize the IR pattern
    and emit the replacement kernel call.
    """
    lhs = op.operands[0]
    rhs = op.operands[1]

    # Choose layout for this operation
    L = d2m.infer_layout(
        op.result,
        tiled=True,
        block_shape=[1, 1],
        grid_shape=[1, 1],
    )

    # Convert inputs and create output
    lhs_d = d2m.to_layout(d2m.from_value(lhs), L)
    rhs_d = d2m.to_layout(d2m.from_value(rhs), L)
    out_d = d2m.empty(L)

    # Call kernel
    my_fused_kernel(lhs_d, rhs_d, out_d, 1, 1, grid=(1, 1))

    return d2m.from_device(out_d)


# ============================================================================
# Step 3: Define test metadata
# ============================================================================

PATTERN_TEST_METADATA = {
    # Unique identifier for your pattern
    "pattern_name": "my_pattern_example",
    # Human-readable description
    "description": "Example pattern showing how to structure tests",
    # ========================================================================
    # LIT Tests: Verify that pattern rewriting works correctly
    # ========================================================================
    "lit_tests": [
        # Positive test case: pattern should match and rewrite
        {
            "name": "my_pattern_positive",
            # MLIR module to test (before pattern application)
            "module_text": """
module {
  func.func @forward(%a: tensor<32x32xf32>, %b: tensor<32x32xf32>)
      -> tensor<32x32xf32> {
    %r = "ttir.mul"(%a, %b) :
        (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    return %r : tensor<32x32xf32>
  }
}
""",
            # FileCheck patterns to verify rewritten IR
            # These are checked in order against the output
            "file_checks": [
                "CHECK-LABEL: func.func @forward",
                "CHECK-NOT:    ttir.mul",  # Original op should be gone
                "CHECK:        d2m.generic",  # Should have d2m kernel call
                "CHECK:        return %{{.*}} : tensor<32x32xf32>",
            ],
            # Optional: description of what this test validates
            "description": "Pattern matches ttir.mul and replaces with kernel",
        },
        # Negative test case: pattern should NOT match (optional)
        # This is useful for testing pattern predicates
        # {
        #     "name": "my_pattern_negative",
        #     "module_text": """...""",
        #     "file_checks": [
        #         "CHECK: ttir.mul",        # Original op should remain
        #         "CHECK-NOT: d2m.generic",  # No kernel should be emitted
        #     ],
        # },
    ],
    # ========================================================================
    # E2E Tests: Verify that kernel produces correct results on device
    # ========================================================================
    "e2e_tests": [
        {
            # Test function name (should start with "test_")
            "name": "test_my_pattern_kernel_on_device",
            # Description shown in pytest output
            "description": "my_fused_kernel matches torch.mul on device",
            # Reference to the kernel function to test
            "kernel_fn": my_fused_kernel,
            # Lambda that generates input tensors
            # Returns a dict mapping parameter names to tensors
            # Torch random seed is set to 0 before this is called
            "input_generator": lambda: {
                "in1": torch.rand(32, 32, dtype=torch.float32),
                "in2": torch.rand(32, 32, dtype=torch.float32),
            },
            # Lambda that computes expected output from inputs
            # Takes the same parameters as input_generator returns
            "reference_fn": lambda in1, in2: in1 * in2,
            # Layout configuration for d2m.Layout()
            # These are passed as kwargs to Layout constructor
            "layout_config": {
                "shape": (32, 32),
                "dtype": d2m.float32,
                "block_shape": [1, 1],
                "grid_shape": [1, 1],
                "tiled": True,
            },
            # Additional arguments passed to kernel
            # These are passed as **kwargs after inputs and output
            "kernel_args": {
                "m_blocks": 1,
                "n_blocks": 1,
                "grid": (1, 1),
            },
        },
        # You can add multiple E2E tests with different configurations
        # {
        #     "name": "test_my_pattern_larger_grid",
        #     "kernel_fn": my_fused_kernel,
        #     "input_generator": lambda: {...},
        #     "reference_fn": lambda ...: ...,
        #     "layout_config": {
        #         "shape": (128, 128),
        #         "grid_shape": [2, 2],  # Larger grid
        #         ...
        #     },
        #     "kernel_args": {
        #         "m_blocks": 2,
        #         "n_blocks": 2,
        #         "grid": (2, 2),
        #     },
        # },
    ],
}


# ============================================================================
# Step 4: Test your pattern
# ============================================================================
#
# Once you've added PATTERN_TEST_METADATA:
#
# 1. Verify discovery finds your pattern:
#    $ cd test/d2m-jit/pattern_tests
#    $ python3 validate_refactoring.py
#
# 2. Run LIT tests (in-process):
#    $ pytest test_lit_generated.py -k "my_pattern"
#
# 3. Run E2E tests (on-device):
#    $ pytest test_e2e_generated.py -k "my_pattern"
#
# 4. Generate standalone LIT files:
#    $ python -m test.d2m_jit.pattern_tests.lit_generator
#
# ============================================================================
