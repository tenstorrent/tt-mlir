# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Operation Registry for TTNN JIT Compiler.

This module provides a complete registration system for operations, including:
- Operation handlers that define how operations are processed
- Argument parsing logic for each operation category
- Automatic attribute management (e.g., ttnn.hoist_generic_via_d2m)
- Easy extensibility for adding new operations

ALL operation-specific logic should be in this file. The compiler should be
completely generic and query this registry to determine how to handle each operation.
"""

from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod
import ttnn_jit._src.supported_ops as supported_ops
from .conversions import ttcore_dtype_from_mlir_dtype


class OpHandler(ABC):
    """
    Base class for operation handlers.

    Each operation category (unary, binary, reduction) or individual operation
    implements this interface to define how it should be processed.
    """

    @abstractmethod
    def parse_arguments(self, op_name: str, arguments: List[str]) -> Dict[str, Any]:
        """
        Parse arguments from levelized graph vertex data.

        Args:
            op_name: Name of the operation
            arguments: List of serialized argument strings from vertex

        Returns:
            Dictionary containing parsed arguments
        """
        pass

    @abstractmethod
    def validate_operands(self, op_name: str, operands: List) -> None:
        """
        Validate that the correct number and type of operands are provided.

        Args:
            op_name: Name of the operation
            operands: List of MLIR operand values

        Raises:
            ValueError: If validation fails
        """
        pass

    @abstractmethod
    def prepare_result_type(
        self, op_name: str, result_type, vertex, compiler_context: Any
    ) -> Any:
        """
        Prepare or modify the result type for this operation.

        Some operations (like reductions) may need to modify the result type
        based on the operation's parameters or output shape.

        Args:
            op_name: Name of the operation
            result_type: Initial MLIR result type
            vertex: The operation vertex (contains output shape info)
            compiler_context: Compiler context for type utilities

        Returns:
            Modified MLIR result type
        """
        pass

    def prepare_operands(
        self,
        op_name: str,
        operands: List,
        parsed_args: Dict[str, Any],
        result_type,
        device,
        compiler_context: Any,
    ) -> List:
        """
        Prepare operands for this operation.

        Some operations (like binary ops with constants) may need to create additional
        operands (e.g., constant tensors) before the operation can be created.

        This method allows each handler to manage its own operand preparation logic,
        making the system extensible for future operation types (e.g., ternary ops).

        Args:
            op_name: Name of the operation
            operands: List of MLIR operand values (may be modified)
            parsed_args: Parsed operation arguments
            result_type: MLIR result type (for creating constants with matching type)
            device: MLIR device value (for creating constants)
            compiler_context: Compiler context for accessing utility methods

        Returns:
            Modified list of operands
        """
        # Default implementation: return operands unchanged
        return operands

    @abstractmethod
    def create_operation(
        self,
        op_name: str,
        result_type,
        operands: List,
        device,
        parsed_args: Dict[str, Any],
        ctx,
    ) -> Any:
        """
        Create the MLIR operation.

        Args:
            op_name: Name of the operation
            result_type: MLIR result type
            operands: List of MLIR operand values
            device: MLIR device value
            parsed_args: Parsed operation arguments
            ctx: MLIR context

        Returns:
            MLIR operation result
        """
        pass

    def add_common_attributes(self, result, ctx):
        """
        Add common attributes that all operations need.

        This includes the "ttnn.hoist_generic_via_d2m" attribute that all
        TTNN operations require.

        Args:
            result: MLIR operation result
            ctx: MLIR context
        """
        from ttmlir.ir import UnitAttr

        result.owner.attributes["ttnn.hoist_generic_via_d2m"] = UnitAttr.get(ctx)


class UnaryOpHandler(OpHandler):
    """Handler for unary operations (single input, single output)."""

    def parse_arguments(self, op_name: str, arguments: List[str]) -> Dict[str, Any]:
        """Unary operations don't have additional parameters we need to parse."""
        return {}

    def validate_operands(self, op_name: str, operands: List) -> None:
        if len(operands) != 1:
            raise ValueError(
                f"Unary operation {op_name} expects 1 operand, got {len(operands)}"
            )

    def prepare_result_type(
        self, op_name: str, result_type, vertex, compiler_context: Any
    ) -> Any:
        # Unary ops typically don't change the type
        return result_type

    def create_operation(
        self,
        op_name: str,
        result_type,
        operands: List,
        device,
        parsed_args: Dict[str, Any],
        ctx,
    ) -> Any:
        from ttmlir.dialects import ttnn

        op_func = getattr(ttnn, op_name)
        result = op_func(result_type, operands[0])
        self.add_common_attributes(result, ctx)
        return result


class BinaryOpHandler(OpHandler):
    """Handler for binary operations (two inputs, single output)."""

    def parse_arguments(self, op_name: str, arguments: List[str]) -> Dict[str, Any]:
        """
        Parse arguments for binary operations.

        Binary operations may have one tensor and one scalar constant.
        The constant may be wrapped in a reference wrapper.
        """
        result = {}

        for arg in arguments:
            if not arg or arg == "nullopt":
                continue

            # Skip tensor specifications
            if arg.startswith("Tensor("):
                continue

            # Try to parse as float constant directly
            if not arg.startswith("["):
                try:
                    constant_value = float(arg)
                    result["constant"] = constant_value
                    break
                except (ValueError, TypeError):
                    pass

        return result

    def validate_operands(self, op_name: str, operands: List) -> None:
        # Will be validated after constant tensor creation if needed
        pass

    def prepare_result_type(
        self, op_name: str, result_type, vertex, compiler_context: Any
    ) -> Any:
        # Binary ops typically don't change the type
        return result_type

    def prepare_operands(
        self,
        op_name: str,
        operands: List,
        parsed_args: Dict[str, Any],
        result_type,
        device,
        compiler_context: Any,
    ) -> List:
        """
        Prepare operands for binary operations.

        Binary operations may have a constant argument that needs to be converted
        to a constant tensor and added to the operand list.
        """
        constant_value = parsed_args.get("constant")
        if constant_value is not None:
            # Create a constant tensor and add it to operands
            constant_tensor = compiler_context._create_constant_tensor(
                constant_value, result_type, device
            )
            operands = operands.copy()  # Don't modify the original list
            operands.append(constant_tensor)

        return operands

    def create_operation(
        self,
        op_name: str,
        result_type,
        operands: List,
        device,
        parsed_args: Dict[str, Any],
        ctx,
    ) -> Any:
        from ttmlir.dialects import ttnn, ttcore

        # Validate operand count (constant was already added if needed)
        if len(operands) != 2:
            raise ValueError(
                f"Binary operation {op_name} expects 2 operands, got {len(operands)}"
            )

        op_func = getattr(ttnn, op_name)
        result = op_func(result_type, operands[0], operands[1])
        self.add_common_attributes(result, ctx)

        # Add dtype as an attribute for binary ops
        dtype_attr = ttcore.ir.DataTypeAttr.get(
            ctx, ttcore_dtype_from_mlir_dtype(result_type.element_type)
        )
        result.owner.attributes["dtype"] = dtype_attr
        return result


class ReductionOpHandler(OpHandler):
    """Handler for reduction operations (reduce along dimensions)."""

    def parse_arguments(self, op_name: str, arguments: List[str]) -> Dict[str, Any]:
        """
        Parse arguments for reduction operations (sum, mean, max, min).

        Arguments format:
        - arguments[0]: Input tensor specification (ignore, comes from in_edges)
        - arguments[1]: dim value (integer or "nullopt")
        - arguments[2]: keep_dim value (0 or 1)
        """
        result = {"dim": None, "keep_dim": False}

        # Extract dim from arguments[1]
        if len(arguments) > 1:
            dim_arg = arguments[1]
            if dim_arg and dim_arg != "nullopt":
                try:
                    dim_val = int(dim_arg)
                    result["dim"] = dim_val
                except (ValueError, TypeError):
                    pass

        # Extract keep_dim from arguments[2]
        if len(arguments) > 2:
            try:
                keep_dim_val = int(arguments[2])
                result["keep_dim"] = bool(keep_dim_val)
            except (ValueError, TypeError):
                pass

        return result

    def validate_operands(self, op_name: str, operands: List) -> None:
        if len(operands) != 1:
            raise ValueError(
                f"Reduction operation {op_name} expects 1 operand, got {len(operands)}"
            )

    def prepare_result_type(
        self, op_name: str, result_type, vertex, compiler_context: Any
    ) -> Any:
        # Reduction operations change the output shape based on which dimensions are reduced.
        # We need to get the correct output type from the vertex's output_shape information.
        return compiler_context._get_output_type_from_vertex(vertex, result_type)

    def create_operation(
        self,
        op_name: str,
        result_type,
        operands: List,
        device,
        parsed_args: Dict[str, Any],
        ctx,
    ) -> Any:
        from ttmlir.dialects import ttnn

        # Extract dim and keep_dim from parsed arguments
        dim_value = parsed_args.get("dim")
        keep_dim_value = parsed_args.get("keep_dim", False)

        op_func = getattr(ttnn, op_name)

        # dim_arg expects a list of dimensions (I32ArrayAttr), not a single int
        # Convert single int to list, or use None for reducing all dimensions
        if dim_value is not None:
            dim_arg = [dim_value]  # Wrap in list for I32ArrayAttr
        else:
            dim_arg = None  # None means reduce all dimensions

        result = op_func(
            result_type, operands[0], keep_dim=keep_dim_value, dim_arg=dim_arg
        )

        self.add_common_attributes(result, ctx)
        return result


class SquareOpHandler(OpHandler):
    """
    Handler for the square operation.

    Square doesn't have a direct MLIR equivalent in the TTNN dialect,
    so we replace it with multiply(x, x).
    Currently we get: AttributeError: module 'ttmlir.dialects.ttnn' has no attribute 'square'
    """

    def parse_arguments(self, op_name: str, arguments: List[str]) -> Dict[str, Any]:
        """Square operations don't have additional parameters."""
        return {}

    def validate_operands(self, op_name: str, operands: List) -> None:
        if len(operands) != 1:
            raise ValueError(f"Square operation expects 1 operand, got {len(operands)}")

    def prepare_result_type(
        self, op_name: str, result_type, vertex, compiler_context: Any
    ) -> Any:
        # Square doesn't change the type
        return result_type

    def create_operation(
        self,
        op_name: str,
        result_type,
        operands: List,
        device,
        parsed_args: Dict[str, Any],
        ctx,
    ) -> Any:
        from ttmlir.dialects import ttnn, ttcore

        # Square is implemented as multiply(x, x)
        op_func = getattr(ttnn, "multiply")
        result = op_func(result_type, operands[0], operands[0])
        self.add_common_attributes(result, ctx)

        # Add dtype attribute (required for binary operations like multiply)
        dtype_attr = ttcore.ir.DataTypeAttr.get(
            ctx, ttcore_dtype_from_mlir_dtype(result_type.element_type)
        )
        result.owner.attributes["dtype"] = dtype_attr
        return result


class OpRegistry:
    """
    Registry for operations and their handlers.

    Operations register themselves with a handler that defines how they should
    be processed. The compiler queries this registry to determine how to handle
    each operation.
    """

    def __init__(self):
        self._handlers: Dict[str, OpHandler] = {}
        self._category_handlers: Dict[str, OpHandler] = {}
        self._op_categories: Dict[str, str] = {}

    def register_op(self, op_name: str, category: str, handler: OpHandler) -> None:
        """
        Register an operation with its handler.

        Args:
            op_name: Name of the operation
            category: Category of the operation (e.g., "unary", "binary")
            handler: Handler instance for this operation
        """
        self._handlers[op_name] = handler
        self._op_categories[op_name] = category

    def register_category(self, category: str, handler: OpHandler) -> None:
        """
        Register a handler for an entire category of operations.

        Args:
            category: Category name (e.g., "unary", "binary", "reduction")
            handler: Handler instance for this category
        """
        self._category_handlers[category] = handler

    def get_handler(self, op_name: str) -> Optional[OpHandler]:
        """
        Get the handler for an operation.

        First checks for op-specific handler, then falls back to category handler.

        Args:
            op_name: Name of the operation

        Returns:
            Handler instance, or None if not found
        """
        # Check for op-specific handler first
        if op_name in self._handlers:
            return self._handlers[op_name]

        # Fall back to category handler
        category = self._op_categories.get(op_name)
        if category and category in self._category_handlers:
            return self._category_handlers[category]

        # Try to infer category from supported_ops
        # This should always work if the op is supported, so we can assert
        if op_name in supported_ops.unary_ops:
            handler = self._category_handlers.get("unary")
            assert (
                handler is not None
            ), f"Unary handler not registered but op {op_name} is in unary_ops"
            return handler
        elif op_name in supported_ops.binary_ops:
            handler = self._category_handlers.get("binary")
            assert (
                handler is not None
            ), f"Binary handler not registered but op {op_name} is in binary_ops"
            return handler
        elif op_name in supported_ops.reduction_ops:
            handler = self._category_handlers.get("reduction")
            assert (
                handler is not None
            ), f"Reduction handler not registered but op {op_name} is in reduction_ops"
            return handler

        # If we reach here, the operation is truly not supported
        return None

    def get_category(self, op_name: str) -> Optional[str]:
        """Get the category of an operation."""
        if op_name in self._op_categories:
            return self._op_categories[op_name]

        # Infer from supported_ops
        if op_name in supported_ops.unary_ops:
            return "unary"
        elif op_name in supported_ops.binary_ops:
            return "binary"
        elif op_name in supported_ops.reduction_ops:
            return "reduction"

        return None


# Global registry instance
_global_registry = OpRegistry()


def get_registry() -> OpRegistry:
    """Get the global operation registry."""
    return _global_registry


def initialize_default_handlers():
    """Initialize the registry with default handlers for standard op categories."""
    registry = get_registry()

    # Register category handlers
    registry.register_category("unary", UnaryOpHandler())
    registry.register_category("binary", BinaryOpHandler())
    registry.register_category("reduction", ReductionOpHandler())

    # Register special operation handlers
    registry.register_op("square", "special", SquareOpHandler())


# Initialize default handlers when module is imported
initialize_default_handlers()
