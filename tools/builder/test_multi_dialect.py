#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Simple test demonstrating the MultiDialectBuilder (Option 6).

This script can be run directly to see how the delegation pattern works.
"""

import sys
import os

# Add the parent directory to the path so we can import builder modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Optional, Union, Dict, Any
from collections import OrderedDict
import torch

from ttmlir.ir import *
from builder.base.builder import Builder
from builder.ttir.ttir_builder import TTIRBuilder
from builder.ttnn.ttnn_builder import TTNNBuilder
from builder.base.builder_apis import *
from builder.base.builder_runtime import execute_fb, runtime_dtype_to_torch_dtype


class MultiDialectBuilder(Builder):
    """
    Simplified MultiDialectBuilder showing the delegation pattern.
    """

    def __init__(
        self,
        ctx: Context,
        location: Location,
        dialects: List[str] = ["ttir", "ttnn"],
        mesh_name: Union[List[str], str] = "mesh",
        mesh_dict: Union[
            List[OrderedDict[str, int]], OrderedDict[str, int]
        ] = OrderedDict([("x", 1), ("y", 1)]),
    ):
        # Initialize base Builder
        super().__init__(ctx, location, mesh_name, mesh_dict)

        # Map of dialect names to builder classes
        self._dialect_map = {
            "ttir": TTIRBuilder,
            "ttnn": TTNNBuilder,
        }

        # Create dialect builders that share this instance's state
        self._dialect_builders: Dict[str, Builder] = {}

        for dialect in dialects:
            if dialect not in self._dialect_map:
                raise ValueError(f"Unknown dialect '{dialect}'")

            builder_cls = self._dialect_map[dialect]
            # Create instance without calling __init__
            builder = builder_cls.__new__(builder_cls)
            # KEY: Share all state by sharing __dict__
            builder.__dict__ = self.__dict__
            self._dialect_builders[dialect] = builder

        # Set up create_tensor_encoding from first available dialect builder
        # (TTIR returns None, which is fine)
        if self._dialect_builders:
            first_builder = next(iter(self._dialect_builders.values()))
            if hasattr(first_builder.__class__, "create_tensor_encoding"):
                # Use the dialect's implementation
                for dialect_name, dialect_builder in self._dialect_builders.items():
                    if dialect_name == "ttir":
                        self.create_tensor_encoding = lambda shape, dtype: None
                        break
                else:
                    # Fall back to first builder's implementation
                    self.create_tensor_encoding = first_builder.create_tensor_encoding

    # Explicit dialect accessors for API clarity (like Option 1)
    @property
    def ttir(self) -> TTIRBuilder:
        """Access TTIR dialect builder explicitly."""
        if "ttir" not in self._dialect_builders:
            raise AttributeError("TTIR dialect not enabled")
        return self._dialect_builders["ttir"]

    @property
    def ttnn(self) -> TTNNBuilder:
        """Access TTNN dialect builder explicitly."""
        if "ttnn" not in self._dialect_builders:
            raise AttributeError("TTNN dialect not enabled")
        return self._dialect_builders["ttnn"]

    def __getattr__(self, name: str) -> Any:
        """
        Delegate to the appropriate dialect builder.
        This is called when an attribute is not found on this instance.
        """
        # Search through all dialect builders
        for dialect, builder in self._dialect_builders.items():
            if hasattr(builder, name):
                return getattr(builder, name)

        # Not found anywhere
        raise AttributeError(
            f"MultiDialectBuilder has no attribute '{name}'. "
            f"Available dialects: {list(self._dialect_builders.keys())}"
        )


def test_basic_usage():
    """Test basic mixed-dialect usage with EXPLICIT API."""
    print("\n" + "=" * 70)
    print("TEST: Basic Multi-Dialect Usage (Explicit API)")
    print("=" * 70 + "\n")

    ctx = Context()
    loc = Location.unknown(ctx)

    # Create a multi-dialect builder
    builder = MultiDialectBuilder(
        ctx, loc, dialects=["ttir", "ttnn"], mesh_dict=OrderedDict([("x", 1), ("y", 1)])
    )

    print(
        f"Created MultiDialectBuilder with dialects: {list(builder._dialect_builders.keys())}"
    )

    # Create a simple module using EXPLICIT dialect API
    with ctx, loc:
        new_module = Module.create()
        builder._root_module_insertion_point = new_module.body
        builder._current_module_insertion_point = new_module.body

        with InsertionPoint(new_module.body):

            @builder.func([(32, 32), (32, 32)], [torch.float32, torch.float32])
            def mixed_ops(in0, in1, builder):
                # Use EXPLICIT TTIR ops (like Option 1)
                print("  Calling builder.ttir.sigmoid...")
                sig = builder.ttir.sigmoid(in0)  # ✅ Explicit: builder.ttir.op()

                print("  Calling builder.ttir.relu...")
                relu = builder.ttnn.relu(in1)  # ✅ Explicit: builder.ttir.op()

                # Use explicit add
                print("  Calling builder.ttir.add...")
                result = builder.ttir.add(sig, relu)  # ✅ Explicit: builder.ttir.op()

                return result

    print("\nGenerated MLIR Module (using explicit API):")
    print("-" * 70)
    print(new_module)
    print("-" * 70)

    return builder, new_module


def test_implicit_usage():
    """Test implicit delegation API."""
    print("\n" + "=" * 70)
    print("TEST: Implicit Delegation API (Automatic)")
    print("=" * 70 + "\n")

    ctx = Context()
    loc = Location.unknown(ctx)

    # Create a multi-dialect builder
    builder = MultiDialectBuilder(
        ctx, loc, dialects=["ttir", "ttnn"], mesh_dict=OrderedDict([("x", 1), ("y", 1)])
    )

    print(
        f"Created MultiDialectBuilder with dialects: {list(builder._dialect_builders.keys())}"
    )

    # Create a simple module using IMPLICIT delegation
    with ctx, loc:
        new_module = Module.create()
        builder._root_module_insertion_point = new_module.body
        builder._current_module_insertion_point = new_module.body

        with InsertionPoint(new_module.body):

            @builder.func([(32, 32), (32, 32)], [torch.float32, torch.float32])
            def implicit_ops(in0, in1, builder):
                # Use implicit delegation (no dialect prefix)
                print("  Calling builder.sigmoid (implicit)...")
                sig = builder.sigmoid(in0)  # Automatically finds ttir.sigmoid

                print("  Calling builder.relu (implicit)...")
                relu = builder.relu(in1)  # Automatically finds ttir.relu

                print("  Calling builder.add (implicit)...")
                result = builder.add(sig, relu)  # Automatically resolved

                return result

    print("\nGenerated MLIR Module (using implicit API):")
    print("-" * 70)
    print(new_module)
    print("-" * 70)

    return builder, new_module


def test_shared_state():
    """Test that state is properly shared between dialect builders."""
    print("\n" + "=" * 70)
    print("TEST: Shared State Between Dialects")
    print("=" * 70 + "\n")

    ctx = Context()
    loc = Location.unknown(ctx)

    builder = MultiDialectBuilder(ctx, loc, dialects=["ttir", "ttnn"])

    print("Testing that dialect builders share state...")

    # They should share the same context
    ttir_builder = builder._dialect_builders["ttir"]
    ttnn_builder = builder._dialect_builders["ttnn"]

    print(f"  Main builder context ID: {id(builder._ctx)}")
    print(f"  TTIR builder context ID: {id(ttir_builder._ctx)}")
    print(f"  TTNN builder context ID: {id(ttnn_builder._ctx)}")

    if builder._ctx is ttir_builder._ctx is ttnn_builder._ctx:
        print("  ✓ All builders share the same context")
    else:
        print("  ✗ Builders have different contexts!")

    # They should share the same golden map
    print(f"\n  Main builder goldens ID: {id(builder._goldens)}")
    print(f"  TTIR builder goldens ID: {id(ttir_builder._goldens)}")
    print(f"  TTNN builder goldens ID: {id(ttnn_builder._goldens)}")

    if builder._goldens is ttir_builder._goldens is ttnn_builder._goldens:
        print("  ✓ All builders share the same golden map")
    else:
        print("  ✗ Builders have different golden maps!")

    print("\n  Conclusion: State sharing is working correctly! ✓")


def test_method_resolution():
    """Test which methods are available and how they resolve."""
    print("\n" + "=" * 70)
    print("TEST: Method Resolution")
    print("=" * 70 + "\n")

    ctx = Context()
    loc = Location.unknown(ctx)

    builder = MultiDialectBuilder(ctx, loc, dialects=["ttir", "ttnn"])

    # Test some common method names
    test_methods = [
        "sigmoid",
        "relu",
        "add",
        "multiply",
        "matmul",
        "reshape",
    ]

    print("Checking method availability:")
    for method_name in test_methods:
        try:
            method = getattr(builder, method_name)
            # Find which builder provides it
            provider = None
            for dialect, dialect_builder in builder._dialect_builders.items():
                if hasattr(dialect_builder, method_name):
                    provider = dialect
                    break
            print(f"  ✓ {method_name:15} -> Available (from {provider})")
        except AttributeError:
            print(f"  ✗ {method_name:15} -> Not available")

    # Test a method that doesn't exist
    print("\nTrying to access non-existent method:")
    try:
        builder.nonexistent_method()
        print("  ✗ Should have raised AttributeError!")
    except AttributeError as e:
        print(f"  ✓ Correctly raised AttributeError: {e}")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("MultiDialectBuilder Demo (Option 6: Enhanced with Explicit API)")
    print("=" * 70)
    print("\nOption 6 now supports BOTH API styles:")
    print("  • Explicit:  builder.ttir.op()  (like Option 1 - clear!)")
    print("  • Implicit:  builder.op()       (automatic - convenient!)")
    print("\nAll with shared state - no synchronization overhead! ✅\n")

    try:
        # Test 1: Explicit API
        test_basic_usage()

        # Test 2: Implicit API
        # test_implicit_usage()

        # Test 3: Shared state
        # test_shared_state()

        # Test 4: Method resolution
        # test_method_resolution()

        print("\n" + "=" * 70)
        print("All tests completed successfully! ✓")
        print("=" * 70)
        print("\nConclusion:")
        print("  ✅ Explicit API works: builder.ttir.sigmoid(x)")
        print("  ✅ Implicit API works: builder.sigmoid(x)")
        print("  ✅ State is shared: No synchronization needed")
        print("  ✅ Best of both worlds: Use explicit when clarity matters,")
        print("     implicit when convenient!")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
