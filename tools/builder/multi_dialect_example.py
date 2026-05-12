# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Example implementation of Option 6: Delegation Pattern for Multi-Dialect Builder

This demonstrates how to create a MultiDialectBuilder that can use ops from
multiple dialects (TTIR, TTNN, StableHLO) in a single module.
"""

from __future__ import annotations
from typing import List, Optional, Union, Dict, Any
from collections import OrderedDict
import torch

from ttmlir.ir import *
from ttmlir.dialects import func

from builder.base.builder import Builder
from builder.ttir.ttir_builder import TTIRBuilder
from builder.ttnn.ttnn_builder import TTNNBuilder
from builder.stablehlo.stablehlo_builder import StableHLOBuilder


class MultiDialectBuilder(Builder):
    """
    A builder that supports operations from multiple dialects within a single module.

    This implementation uses a delegation pattern where:
    1. All dialect builders share the same state (context, goldens, etc.)
    2. Method calls are automatically routed to the appropriate dialect builder
    3. The API remains clean and transparent to users
    """

    def __init__(
        self,
        ctx: Context,
        location: Location,
        dialects: List[str] = ["ttir", "ttnn", "stablehlo"],
        mesh_name: Union[List[str], str] = "mesh",
        mesh_dict: Union[
            List[OrderedDict[str, int]], OrderedDict[str, int]
        ] = OrderedDict([("x", 1), ("y", 1)]),
        deallocate_goldens: bool = False,
        deallocated_goldens_dir: Optional[str] = "./deallocated_goldens",
    ):
        """
        Initialize MultiDialectBuilder with specified dialects.

        Parameters
        ----------
        ctx : Context
            MLIR context
        location : Location
            Default location for operations
        dialects : List[str]
            List of dialects to enable (e.g., ["ttir", "ttnn", "stablehlo"])
        mesh_name : Union[List[str], str]
            Mesh name(s) for distributed operations
        mesh_dict : Union[List[OrderedDict[str, int]], OrderedDict[str, int]]
            Mesh shape specification
        deallocate_goldens : bool
            Whether to deallocate golden tensors to disk
        deallocated_goldens_dir : Optional[str]
            Directory for deallocated goldens
        """
        # Initialize base Builder with shared state
        super().__init__(
            ctx,
            location,
            mesh_name,
            mesh_dict,
            deallocate_goldens=deallocate_goldens,
            deallocated_goldens_dir=deallocated_goldens_dir,
        )

        # Create dialect builders that share this instance's state
        self._dialect_builders: Dict[str, Builder] = {}
        self._dialect_map = {
            "ttir": TTIRBuilder,
            "ttnn": TTNNBuilder,
            "stablehlo": StableHLOBuilder,
        }

        # Initialize each requested dialect builder
        for dialect in dialects:
            if dialect not in self._dialect_map:
                raise ValueError(
                    f"Unknown dialect '{dialect}'. "
                    f"Available dialects: {list(self._dialect_map.keys())}"
                )

            builder_cls = self._dialect_map[dialect]
            # Create instance without calling __init__
            builder = builder_cls.__new__(builder_cls)
            # Share all state with this instance
            builder.__dict__ = self.__dict__
            self._dialect_builders[dialect] = builder

        # Set up create_tensor_encoding from available dialect builders
        # Priority: TTIR (returns None) > TTNN > StableHLO
        if "ttir" in self._dialect_builders:
            self.create_tensor_encoding = lambda shape, dtype: None
        elif "ttnn" in self._dialect_builders:
            ttnn_builder = self._dialect_builders["ttnn"]
            if hasattr(ttnn_builder, "_create_tensor_encoding"):
                self.create_tensor_encoding = ttnn_builder._create_tensor_encoding
        elif "stablehlo" in self._dialect_builders:
            self.create_tensor_encoding = lambda shape, dtype: None
        else:
            # Fallback
            self.create_tensor_encoding = lambda shape, dtype: None

        # Keep track of which methods belong to which dialect for debugging
        self._method_to_dialect: Dict[str, str] = {}
        for dialect, builder in self._dialect_builders.items():
            for attr_name in dir(builder):
                if not attr_name.startswith("_") and callable(
                    getattr(builder, attr_name)
                ):
                    # Track which dialect provides this method
                    if attr_name not in self._method_to_dialect:
                        self._method_to_dialect[attr_name] = dialect

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

    @property
    def stablehlo(self) -> StableHLOBuilder:
        """Access StableHLO dialect builder explicitly."""
        if "stablehlo" not in self._dialect_builders:
            raise AttributeError("StableHLO dialect not enabled")
        return self._dialect_builders["stablehlo"]

    def __getattr__(self, name: str) -> Any:
        """
        Delegate attribute access to appropriate dialect builder.

        This method is called when an attribute is not found on the instance.
        It searches through all dialect builders to find the requested method.

        Note: This provides automatic delegation for convenience, but you can
        also use explicit dialect access (builder.ttir.op(), builder.ttnn.op())
        for better API clarity.
        """
        # Try to find the method in dialect builders
        for dialect, builder in self._dialect_builders.items():
            if hasattr(builder, name):
                attr = getattr(builder, name)
                # If it's a method, return it bound to the shared state
                return attr

        # If not found in any dialect builder, raise AttributeError
        raise AttributeError(
            f"MultiDialectBuilder has no attribute '{name}'. "
            f"Available dialects: {list(self._dialect_builders.keys())}"
        )

    def get_method_dialect(self, method_name: str) -> Optional[str]:
        """
        Get which dialect provides a specific method.

        Useful for debugging and understanding method resolution.
        """
        return self._method_to_dialect.get(method_name)

    def list_methods_by_dialect(self) -> Dict[str, List[str]]:
        """
        Return a dictionary mapping dialects to their available methods.

        Useful for debugging and documentation.
        """
        result = {}
        for dialect, builder in self._dialect_builders.items():
            methods = [
                name
                for name in dir(builder)
                if not name.startswith("_") and callable(getattr(builder, name))
            ]
            result[dialect] = sorted(methods)
        return result


# ============================================================================
# Usage Example 1: Explicit Dialect API (like Option 1)
# ============================================================================


def example_explicit_dialect_api():
    """
    Example: Using explicit dialect API for clarity (builder.ttir.op(), builder.ttnn.op()).
    This provides the same API clarity as Option 1, but with shared state benefits.
    """
    ctx = Context()
    loc = Location.unknown(ctx)

    multi_builder = MultiDialectBuilder(
        ctx, loc, dialects=["ttir", "ttnn"], mesh_dict=OrderedDict([("x", 1), ("y", 1)])
    )

    def module(builder: MultiDialectBuilder):
        @builder.func([(32, 32), (32, 32)], [torch.float32, torch.float32])
        def explicit_function(in0, in1, builder: MultiDialectBuilder):
            # Explicit TTIR operations - crystal clear which dialect!
            ttir_result = builder.ttir.sigmoid(in0)  # ✅ Explicitly TTIR
            ttir_result2 = builder.ttir.relu(in1)  # ✅ Explicitly TTIR

            # Explicit TTNN operation - no ambiguity!
            ttnn_result = builder.ttnn.multiply(
                ttir_result, ttir_result2
            )  # ✅ Explicitly TTNN

            return ttnn_result

    # Build the module
    with ctx, loc:
        new_module = Module.create()
        multi_builder._root_module_insertion_point = new_module.body
        multi_builder._current_module_insertion_point = new_module.body

        with InsertionPoint(new_module.body):
            module(multi_builder)

    print("=" * 70)
    print("Explicit Dialect API (builder.ttir.op(), builder.ttnn.op()):")
    print("=" * 70)
    print(new_module)
    print()

    return new_module, multi_builder


# ============================================================================
# Usage Example 2: Implicit Delegation API (for convenience)
# ============================================================================


def example_implicit_delegation_api():
    """
    Example: Using implicit delegation for convenience (builder.op()).
    Automatically finds the right dialect, but less explicit.
    """
    ctx = Context()
    loc = Location.unknown(ctx)

    multi_builder = MultiDialectBuilder(
        ctx, loc, dialects=["ttir", "ttnn"], mesh_dict=OrderedDict([("x", 1), ("y", 1)])
    )

    def module(builder: MultiDialectBuilder):
        @builder.func([(32, 32), (32, 32)], [torch.float32, torch.float32])
        def implicit_function(in0, in1, builder: MultiDialectBuilder):
            # Implicit delegation - shorter but less explicit
            result1 = builder.sigmoid(in0)  # Automatically finds ttir.sigmoid
            result2 = builder.relu(in1)  # Automatically finds ttir.relu
            result3 = builder.multiply(
                result1, result2
            )  # Finds multiply in available dialects

            return result3

    # Build the module
    with ctx, loc:
        new_module = Module.create()
        multi_builder._root_module_insertion_point = new_module.body
        multi_builder._current_module_insertion_point = new_module.body

        with InsertionPoint(new_module.body):
            module(multi_builder)

    print("=" * 70)
    print("Implicit Delegation API (builder.op()) - Automatic Resolution:")
    print("=" * 70)
    print(new_module)
    print()

    return new_module, multi_builder


# ============================================================================
# Usage Example 3: Mixed API Style (Best of Both Worlds)
# ============================================================================


def example_mixed_api_style():
    """
    Example: Mixing explicit and implicit APIs based on needs.
    Use explicit when clarity matters, implicit when convenient.
    """
    ctx = Context()
    loc = Location.unknown(ctx)

    multi_builder = MultiDialectBuilder(
        ctx, loc, dialects=["ttir", "ttnn"], mesh_dict=OrderedDict([("x", 1), ("y", 1)])
    )

    def module(builder: MultiDialectBuilder):
        @builder.func([(32, 32), (32, 32)], [torch.float32, torch.float32])
        def mixed_style_function(in0, in1, builder: MultiDialectBuilder):
            # Use EXPLICIT when clarity is important
            # (e.g., when multiple dialects have the same op name)
            x = builder.ttir.sigmoid(in0)  # ✅ Clear: using TTIR sigmoid
            y = builder.ttir.relu(in1)  # ✅ Clear: using TTIR relu

            # Use IMPLICIT when there's no ambiguity
            # (shorter, more convenient)
            z = builder.add(x, y)  # OK: only one 'add' available, or don't care which

            # Use EXPLICIT when semantic differences matter
            result = builder.ttnn.multiply(z, x)  # ✅ Clear: TTNN-specific semantics

            return result

    # Build the module
    with ctx, loc:
        new_module = Module.create()
        multi_builder._root_module_insertion_point = new_module.body
        multi_builder._current_module_insertion_point = new_module.body

        with InsertionPoint(new_module.body):
            module(multi_builder)

    print("=" * 70)
    print("Mixed API Style (Explicit + Implicit):")
    print("=" * 70)
    print(new_module)
    print()
    print("Key benefits:")
    print("  ✅ Explicit dialect access when clarity matters")
    print("  ✅ Implicit delegation when convenient")
    print("  ✅ Shared state (no synchronization overhead)")
    print()

    return new_module, multi_builder


# ============================================================================
# Usage Example 4: StableHLO to TTIR Conversion Pattern
# ============================================================================


def example_stablehlo_to_ttir():
    """
    Example: Create a module that uses StableHLO ops that might be
    lowered to TTIR ops within the same module.
    """
    ctx = Context()
    loc = Location.unknown(ctx)

    multi_builder = MultiDialectBuilder(
        ctx,
        loc,
        dialects=["stablehlo", "ttir"],
        mesh_dict=OrderedDict([("x", 2), ("y", 2)]),
    )

    def module(builder: MultiDialectBuilder):
        @builder.func([(64, 64)], [torch.float32])
        def hybrid_function(input_tensor, builder: MultiDialectBuilder):
            # Explicit StableHLO operation
            stablehlo_result = builder.stablehlo.abs(input_tensor)  # ✅ Clear: StableHLO

            # Explicit TTIR operation
            ttir_result = builder.ttir.sigmoid(stablehlo_result)  # ✅ Clear: TTIR

            return ttir_result

    with ctx, loc:
        new_module = Module.create()
        multi_builder._root_module_insertion_point = new_module.body
        multi_builder._current_module_insertion_point = new_module.body

        with InsertionPoint(new_module.body):
            module(multi_builder)

    print("=" * 70)
    print("Mixed StableHLO/TTIR Module (Explicit API):")
    print("=" * 70)
    print(new_module)
    print()

    return new_module, multi_builder


# ============================================================================
# Usage Example 5: Introspection
# ============================================================================


def example_introspection():
    """
    Example: Demonstrate introspection capabilities of MultiDialectBuilder.
    """
    ctx = Context()
    loc = Location.unknown(ctx)

    multi_builder = MultiDialectBuilder(
        ctx, loc, dialects=["ttir", "ttnn", "stablehlo"]
    )

    print("=" * 70)
    print("MultiDialectBuilder Introspection:")
    print("=" * 70)

    # List available dialects
    print(f"\nEnabled dialects: {list(multi_builder._dialect_builders.keys())}")

    # Show that explicit access works
    print("\nExplicit dialect access:")
    print(f"  builder.ttir type: {type(multi_builder.ttir).__name__}")
    print(f"  builder.ttnn type: {type(multi_builder.ttnn).__name__}")
    print(f"  builder.stablehlo type: {type(multi_builder.stablehlo).__name__}")

    # Check which dialect provides specific methods
    example_methods = ["sigmoid", "add", "multiply", "abs", "reshape"]
    print("\nMethod resolution (for implicit API):")
    for method in example_methods:
        dialect = multi_builder.get_method_dialect(method)
        if dialect:
            print(f"  {method:15} -> {dialect}")
        else:
            print(f"  {method:15} -> Not found")

    # List all methods by dialect (showing just a few for brevity)
    print("\nSample methods by dialect:")
    methods_by_dialect = multi_builder.list_methods_by_dialect()
    for dialect, methods in methods_by_dialect.items():
        print(f"  {dialect}: {len(methods)} methods")
        # Show first 5 methods as example
        for method in methods[:5]:
            print(f"    - {method}")
        if len(methods) > 5:
            print(f"    ... and {len(methods) - 5} more")

    print()


# ============================================================================
# Usage Example 6: Practical Pattern - Building a Complete Network
# ============================================================================


def example_practical_network():
    """
    Example: A more realistic use case building a small neural network
    that might use operations from multiple dialects.
    Uses explicit API for clarity.
    """
    ctx = Context()
    loc = Location.unknown(ctx)

    multi_builder = MultiDialectBuilder(
        ctx, loc, dialects=["ttir", "ttnn"], mesh_dict=OrderedDict([("x", 1), ("y", 1)])
    )

    def simple_mlp(builder: MultiDialectBuilder):
        """Build a simple MLP with explicit dialect ops."""

        @builder.func(
            [(128, 784), (784, 256), (256,)],
            [torch.float32, torch.float32, torch.float32],
        )
        def forward(x, w1, b1, builder: MultiDialectBuilder):
            # Layer 1: Use explicit TTIR operations
            h1 = builder.ttir.matmul(x, w1)  # ✅ Explicitly TTIR matmul
            h1 = builder.ttir.add(h1, b1)  # ✅ Explicitly TTIR add

            # Activation: Explicit TTIR
            h1 = builder.ttir.relu(h1)  # ✅ Explicitly TTIR relu

            # Could continue with TTNN ops if needed
            # output = builder.ttnn.some_specialized_op(h1)

            return h1

    with ctx, loc:
        new_module = Module.create()
        multi_builder._root_module_insertion_point = new_module.body
        multi_builder._current_module_insertion_point = new_module.body

        with InsertionPoint(new_module.body):
            simple_mlp(multi_builder)

    print("=" * 70)
    print("Practical MLP Network (Explicit Dialect API):")
    print("=" * 70)
    print(new_module)
    print()

    return new_module, multi_builder


# ============================================================================
# Main - Run Examples
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("MultiDialectBuilder Examples (Option 6: Enhanced with Explicit API)")
    print("=" * 70 + "\n")

    # Example 1: Explicit API (like Option 1)
    print("EXAMPLE 1: Explicit Dialect API (builder.ttir.op())")
    print("-" * 70)
    try:
        example_explicit_dialect_api()
    except Exception as e:
        print(f"Error: {e}\n")

    # Example 2: Implicit API (automatic delegation)
    print("EXAMPLE 2: Implicit Delegation API (builder.op())")
    print("-" * 70)
    try:
        example_implicit_delegation_api()
    except Exception as e:
        print(f"Error: {e}\n")

    # Example 3: Mixed style
    print("EXAMPLE 3: Mixed API Style (Best of Both)")
    print("-" * 70)
    try:
        example_mixed_api_style()
    except Exception as e:
        print(f"Error: {e}\n")

    # Example 4: StableHLO to TTIR
    print("EXAMPLE 4: StableHLO to TTIR Hybrid")
    print("-" * 70)
    try:
        example_stablehlo_to_ttir()
    except Exception as e:
        print(f"Error: {e}\n")

    # Example 5: Introspection
    print("EXAMPLE 5: Introspection and Debugging")
    print("-" * 70)
    try:
        example_introspection()
    except Exception as e:
        print(f"Error: {e}\n")

    # Example 6: Practical network
    print("EXAMPLE 6: Practical Neural Network")
    print("-" * 70)
    try:
        example_practical_network()
    except Exception as e:
        print(f"Error: {e}\n")

    print("=" * 70)
    print("Examples Complete")
    print("=" * 70)
    print("\nKey Takeaway: Option 6 supports BOTH API styles:")
    print("  • Explicit:  builder.ttir.op()  (like Option 1)")
    print("  • Implicit:  builder.op()       (automatic)")
    print("  • Mixed:     Use both as needed!")
    print("\nAll with shared state - no synchronization overhead! ✅")
    # Mix in TTIR-specific operations
    ttir_result = builder.sigmoid(stablehlo_result)  # ttir.sigmoid

    return ttir_result

    with ctx, loc:
        new_module = Module.create()
        multi_builder._root_module_insertion_point = new_module.body
        multi_builder._current_module_insertion_point = new_module.body

        with InsertionPoint(new_module.body):
            module(multi_builder)

    print("=" * 70)
    print("Mixed StableHLO/TTIR Module:")
    print("=" * 70)
    print(new_module)
    print()

    return new_module, multi_builder


# ============================================================================
# Usage Example 3: Debugging and Introspection
# ============================================================================


def example_introspection():
    """
    Example: Demonstrate introspection capabilities of MultiDialectBuilder.
    """
    ctx = Context()
    loc = Location.unknown(ctx)

    multi_builder = MultiDialectBuilder(
        ctx, loc, dialects=["ttir", "ttnn", "stablehlo"]
    )

    print("=" * 70)
    print("MultiDialectBuilder Introspection:")
    print("=" * 70)

    # List available dialects
    print(f"\nEnabled dialects: {list(multi_builder._dialect_builders.keys())}")

    # Check which dialect provides specific methods
    example_methods = ["sigmoid", "add", "multiply", "abs", "reshape"]
    print("\nMethod resolution:")
    for method in example_methods:
        dialect = multi_builder.get_method_dialect(method)
        if dialect:
            print(f"  {method:15} -> {dialect}")
        else:
            print(f"  {method:15} -> Not found")

    # List all methods by dialect (showing just a few for brevity)
    print("\nSample methods by dialect:")
    methods_by_dialect = multi_builder.list_methods_by_dialect()
    for dialect, methods in methods_by_dialect.items():
        print(f"  {dialect}: {len(methods)} methods")
        # Show first 5 methods as example
        for method in methods[:5]:
            print(f"    - {method}")
        if len(methods) > 5:
            print(f"    ... and {len(methods) - 5} more")

    print()


# ============================================================================
# Usage Example 4: Practical Pattern - Building a Complete Network
# ============================================================================


def example_practical_network():
    """
    Example: A more realistic use case building a small neural network
    that might use operations from multiple dialects.
    """
    ctx = Context()
    loc = Location.unknown(ctx)

    multi_builder = MultiDialectBuilder(
        ctx, loc, dialects=["ttir", "ttnn"], mesh_dict=OrderedDict([("x", 1), ("y", 1)])
    )

    def simple_mlp(builder: MultiDialectBuilder):
        """Build a simple MLP with mixed dialect ops."""

        @builder.func(
            [(128, 784), (784, 256), (256,)],
            [torch.float32, torch.float32, torch.float32],
        )
        def forward(x, w1, b1, builder: MultiDialectBuilder):
            # Layer 1: Matrix multiply + bias add
            h1 = builder.matmul(x, w1)  # Shape: (128, 256)
            h1 = builder.add(h1, b1)  # Add bias

            # Activation
            h1 = builder.relu(h1)

            # Could continue with more layers...
            return h1

    with ctx, loc:
        new_module = Module.create()
        multi_builder._root_module_insertion_point = new_module.body
        multi_builder._current_module_insertion_point = new_module.body

        with InsertionPoint(new_module.body):
            simple_mlp(multi_builder)

    print("=" * 70)
    print("Practical MLP Network:")
    print("=" * 70)
    print(new_module)
    print()

    return new_module, multi_builder


# ============================================================================
# Main - Run Examples
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("MultiDialectBuilder Examples (Option 6: Delegation Pattern)")
    print("=" * 70 + "\n")

    # Example 1: Basic mixed TTIR/TTNN
    print("EXAMPLE 1: Basic Mixed TTIR/TTNN Usage")
    print("-" * 70)
    try:
        example_mixed_ttir_ttnn()
    except Exception as e:
        print(f"Error: {e}\n")

    # Example 2: StableHLO to TTIR
    print("EXAMPLE 2: StableHLO to TTIR Hybrid")
    print("-" * 70)
    try:
        example_stablehlo_to_ttir()
    except Exception as e:
        print(f"Error: {e}\n")

    # Example 3: Introspection
    print("EXAMPLE 3: Introspection and Debugging")
    print("-" * 70)
    try:
        example_introspection()
    except Exception as e:
        print(f"Error: {e}\n")

    # Example 4: Practical network
    print("EXAMPLE 4: Practical Neural Network")
    print("-" * 70)
    try:
        example_practical_network()
    except Exception as e:
        print(f"Error: {e}\n")

    print("=" * 70)
    print("Examples Complete")
    print("=" * 70)
