#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Integration test for MultiDialectBuilder.

Tests that the integrated MultiDialectBuilder works correctly via:
1. Direct import from builder module
2. build_module() API with builder_type="multi"
"""

import sys
from collections import OrderedDict
import torch

from ttmlir.ir import *
from builder import MultiDialectBuilder, build_module


def test_direct_import():
    """Test direct import and usage of MultiDialectBuilder."""
    print("=" * 70)
    print("TEST 1: Direct Import")
    print("=" * 70)

    ctx = Context()
    loc = Location.unknown(ctx)

    # Create builder directly
    builder = MultiDialectBuilder(
        ctx, loc, dialects=["ttir", "ttnn"], mesh_dict=OrderedDict([("x", 1), ("y", 1)])
    )

    print(f"✓ MultiDialectBuilder imported successfully")
    print(f"✓ Enabled dialects: {builder.list_enabled_dialects()}")

    # Test explicit API
    print(f"✓ Explicit API: builder.ttir = {type(builder.ttir).__name__}")
    print(f"✓ Explicit API: builder.ttnn = {type(builder.ttnn).__name__}")

    print("✓ TEST 1 PASSED\n")


def test_build_module_api():
    """Test MultiDialectBuilder via build_module() API."""
    print("=" * 70)
    print("TEST 2: build_module() API")
    print("=" * 70)

    def my_module(builder):
        @builder.func([(32, 32), (32, 32)], [torch.float32, torch.float32])
        def forward(in0, in1, builder):
            # Use explicit API
            x = builder.ttir.sigmoid(in0)
            y = builder.ttir.relu(in1)
            z = builder.ttir.add(x, y)
            return z

    # Build with multi-dialect builder
    module, builder = build_module(
        my_module, builder_type="multi", dialects=["ttir", "ttnn"]
    )

    print(f"✓ Module built successfully")
    print(f"✓ Builder type: {type(builder).__name__}")
    print(f"✓ Enabled dialects: {builder.list_enabled_dialects()}")
    print("✓ TEST 2 PASSED\n")


def test_implicit_api():
    """Test implicit API delegation."""
    print("=" * 70)
    print("TEST 3: Implicit API Delegation")
    print("=" * 70)

    def my_module(builder):
        @builder.func([(32, 32)], [torch.float32])
        def forward(input, builder):
            # Use implicit API (automatic delegation)
            x = builder.sigmoid(input)
            y = builder.relu(x)
            return y

    module, builder = build_module(
        my_module, builder_type="multi", dialects=["ttir", "ttnn"]
    )

    print(f"✓ Implicit API works")
    print(f"✓ sigmoid resolved to: {builder.get_method_dialect('sigmoid')}")
    print(f"✓ relu resolved to: {builder.get_method_dialect('relu')}")
    print("✓ TEST 3 PASSED\n")


def test_mixed_api():
    """Test mixing explicit and implicit APIs."""
    print("=" * 70)
    print("TEST 4: Mixed API (Explicit + Implicit)")
    print("=" * 70)

    def my_module(builder):
        @builder.func([(32, 32), (32, 32)], [torch.float32, torch.float32])
        def forward(in0, in1, builder):
            # Explicit when clarity matters
            x = builder.ttir.sigmoid(in0)

            # Implicit when convenient
            y = builder.relu(in1)

            # Explicit for final op
            z = builder.ttir.add(x, y)
            return z

    module, builder = build_module(
        my_module, builder_type="multi", dialects=["ttir", "ttnn"]
    )

    print(f"✓ Mixed API style works")
    print("✓ TEST 4 PASSED\n")


def main():
    """Run all integration tests."""
    print("\n" + "=" * 70)
    print("MultiDialectBuilder Integration Tests")
    print("=" * 70 + "\n")

    try:
        test_direct_import()
        test_build_module_api()
        test_implicit_api()
        test_mixed_api()

        print("=" * 70)
        print("ALL TESTS PASSED ✓")
        print("=" * 70)
        print("\nMultiDialectBuilder is successfully integrated!")
        print("You can now use:")
        print("  • from builder import MultiDialectBuilder")
        print("  • build_module(..., builder_type='multi', dialects=[...])")
        print("=" * 70 + "\n")

        return 0

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
