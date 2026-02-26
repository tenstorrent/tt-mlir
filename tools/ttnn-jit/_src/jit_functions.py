# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from functools import partial
import math

from ttmlir.dialects import ttir, arith
from ttmlir.ir import (
    InsertionPoint,
    Location,
    RankedTensorType,
    StringAttr,
    F32Type,
    F64Type,
    IntegerType,
    IntegerAttr,
    FloatAttr,
    DenseElementsAttr,
    BF16Type,
)

from ttnn_jit._src.supported_ops import (
    unary_ops,
    binary_ops,
    reduction_ops,
    tm_ops,
    get_ttir_name,
    TTIR_NAME_MAP,
)
from ttnn_jit._src.tensor_translator import (
    create_default_dram_interleaved_layout,
    create_output_tensor,
)

"""
Set this to True to create encoding for intermediate tensors (and the final output tensor), False otherwise.
Encodings are created based on the following logic:
1. For Unary and Binary operations, the encoding is the same as the first input operand.
2. For Matmul operation, the encoding is L1 Block Sharded layout.
3. For all other operations named hereafter, the encoding is a default DRAM interleaved layout:
   - Reductions (Sum, Mean, Max, Min)
   - TMs (Permute, Transpose, Reshape)
"""
CREATE_INTERMEDIATE_LAYOUT = True


class ResultWrapper:
    """Wrapper to track MLIR operation results."""

    def __init__(self, mlir_value, original_value=None):
        self.mlir_value = mlir_value
        self.original_value = original_value


class BaseOpHandler(ABC):
    """Abstract base class for all operation handlers."""

    def __init__(self, jit_ctx):
        self.jit_ctx = jit_ctx

    @staticmethod
    def _resolve_operand(arg, arg_index, jit_ctx):
        """
        Resolve a Python argument to its corresponding MLIR operand.

        Resolution order:
        1. Check if arg is a scalar constant (int, float, bool) - return as-is
        2. Check if arg is a ResultWrapper (has mlir_value attribute)
        3. Check if arg is in value_map (tracked intermediate result)
        4. Fallback to function block argument (input argument)

        Args:
            arg: Python argument (tensor, scalar, or ResultWrapper)
            arg_index: Index of the argument in the function signature
            jit_ctx: JIT context containing value_map, func_bb, etc.

        Returns:
            MLIR Value corresponding to the argument, or scalar value as-is
        """
        if arg is None:
            raise ValueError(f"Argument at index {arg_index} is None")

        # Check if arg is a scalar constant (int, float, bool)
        # These are passed as-is to the operation handlers
        if isinstance(arg, (int, float, bool)):
            return arg

        # Check if arg is a ResultWrapper (intermediate result with mlir_value)
        if hasattr(arg, "mlir_value"):
            return arg.mlir_value

        # Check if arg is in value_map (tracked intermediate result)
        arg_id = id(arg)
        if arg_id in jit_ctx.value_map:
            return jit_ctx.value_map[arg_id]

        # Fallback to function block arguments (input arguments)
        if arg_index < len(jit_ctx.func_bb.arguments):
            return jit_ctx.func_bb.arguments[arg_index]

        raise IndexError(
            f"Argument index {arg_index} out of range. "
            f"Function has {len(jit_ctx.func_bb.arguments)} arguments."
        )

    def _get_operands(self, args):
        """Get MLIR operands from Python arguments."""
        operands = []
        for i, arg in enumerate(args):
            if arg is None:
                raise ValueError(f"Argument at index {i} is None")
            operand = self._resolve_operand(arg, i, self.jit_ctx)
            operands.append(operand)
        return operands

    def _create_result_wrapper(self, mlir_value, original_value=None):
        """Create a ResultWrapper for the operation result."""
        return ResultWrapper(mlir_value, original_value)

    def _store_result(self, arg_id, mlir_value):
        """Store result in value_map if not an original function argument."""
        if arg_id not in self.jit_ctx.func_arg_ids:
            self.jit_ctx.value_map[arg_id] = mlir_value

    def _finalize_result(self, op_result, args):
        """
        Store the operation result and create a ResultWrapper.

        This is the common finalization logic for all operation handlers.

        Args:
            op_result: The MLIR operation result value
            args: The original arguments passed to the operation

        Returns:
            ResultWrapper containing the operation result
        """
        if args:
            arg_id = id(args[0])
            self._store_result(arg_id, op_result)
        return self._create_result_wrapper(op_result, args[0] if args else None)

    def _create_scalar_tensor_constant(self, scalar_value, reference_tensor_operand):
        """
        Create a scalar constant tensor using ttir.full operation.
        This creates a tensor filled with the scalar value that will be broadcast to match
        the other operand's shape. Unlike the reference tensor which might have a complex
        layout (sharded, etc.), the constant uses a simple no-layout tensor that can be
        efficiently broadcast during lowering.

        Args:
            scalar_value: Python scalar (int, float, or bool)
            reference_tensor_operand: MLIR tensor operand to match shape/type from

        Returns:
            MLIR Value representing the constant tensor
        """
        # Get shape and element type from reference tensor
        ref_type = reference_tensor_operand.type
        shape = list(ref_type.shape)
        element_type = ref_type.element_type

        with InsertionPoint(self.jit_ctx.func_bb), Location.unknown(self.jit_ctx.ctx):
            # Create output tensor type without layout encoding
            # The layout will be inferred/added by TTIR lowering passes
            output_type = RankedTensorType.get(shape, element_type)

            # Convert scalar to MLIR Attribute (f32 for floats, i32 for ints)
            # ttir.full expects the fill_value as f32 or i32 attribute
            if isinstance(scalar_value, bool):
                fill_value_attr = IntegerAttr.get(
                    IntegerType.get_signless(32, self.jit_ctx.ctx), int(scalar_value)
                )
            elif isinstance(scalar_value, int):
                fill_value_attr = IntegerAttr.get(
                    IntegerType.get_signless(32, self.jit_ctx.ctx), scalar_value
                )
            elif isinstance(scalar_value, float):
                fill_value_attr = FloatAttr.get(
                    F32Type.get(self.jit_ctx.ctx), scalar_value
                )
            else:
                raise ValueError(f"Unsupported scalar type: {type(scalar_value)}")

            # Use ttir.full to create a tensor filled with the scalar value
            # Note: ttir.full expects shape as DenseI32ArrayAttr and fill_value as Attribute
            from ttmlir.ir import DenseI32ArrayAttr

            shape_attr = DenseI32ArrayAttr.get(shape, self.jit_ctx.ctx)

            return ttir.full(output_type, shape_attr, fill_value_attr)

    def _default_dram_interleaved_layout(self, element_type, new_shape):
        """
        Create a default DRAM interleaved layout for a given shape.
        """
        return (
            create_default_dram_interleaved_layout(
                self.jit_ctx.ctx, new_shape, element_type
            )
            if CREATE_INTERMEDIATE_LAYOUT
            else None
        )

    def _normalize_dim(self, dim, rank):
        """Normalize negative dimension index to positive."""
        if dim < 0:
            dim = rank + dim
        assert (
            0 <= dim < rank
        ), f"Dimension {dim} out of range for tensor with rank {rank}"
        return dim

    @abstractmethod
    def create_operation(self, *args, **kwargs):
        """Create the MLIR operation. Must be implemented by subclasses."""
        pass


class UnaryOpHandler(BaseOpHandler):
    """Handler for unary operations."""

    def __init__(self, jit_ctx, op_name):
        super().__init__(jit_ctx)
        self.op_name = op_name
        if op_name not in unary_ops:
            raise ValueError(f"Unknown unary operation: {op_name}")

    def _infer_output_layout(self, operand):
        """Infer output layout for unary ops - preserves input encoding."""
        return operand.type.encoding if CREATE_INTERMEDIATE_LAYOUT else None

    def _infer_result_type(self, operand):
        """Infer result type from operand, preserving encoding (layout unchanged)."""
        element_type = operand.type.element_type
        shape = list(operand.type.shape)
        encoding = self._infer_output_layout(operand)

        with Location.unknown(self.jit_ctx.ctx):
            return RankedTensorType.get(shape, element_type, encoding)

    def create_operation(self, *args, **kwargs):
        """Create unary operation."""
        if len(args) < 1:
            raise ValueError(f"{self.op_name} requires at least 1 argument")

        operands = self._get_operands(args)
        operand = operands[0]

        # Check if operand is a scalar - unary ops require tensor operands
        if isinstance(operand, (int, float, bool)):
            raise ValueError(
                f"{self.op_name} requires a tensor operand, got scalar: {operand}"
            )

        # Infer result type (no layout for intermediate results)
        result_type = self._infer_result_type(operand)

        # Get the TTIR operation constructor
        op_constructor = getattr(ttir, get_ttir_name(self.op_name))

        with InsertionPoint(self.jit_ctx.func_bb), Location.unknown(self.jit_ctx.ctx):
            op_result = op_constructor(result=result_type, input=operand)

        return self._finalize_result(op_result, args)


class BinaryOpHandler(BaseOpHandler):
    """Handler for binary operations."""

    def __init__(self, jit_ctx, op_name):
        super().__init__(jit_ctx)
        self.op_name = op_name
        if op_name not in binary_ops:
            raise ValueError(f"Unknown binary operation: {op_name}")

    def _infer_output_layout(self, operand0, operand1):
        """Infer output layout for binary ops - preserves first operand's encoding."""
        return operand0.type.encoding if CREATE_INTERMEDIATE_LAYOUT else None

    def _infer_result_type(self, operand0, operand1):
        """Infer result type from operands, preserving encoding from first operand."""
        element_type = operand0.type.element_type
        shape = list(operand0.type.shape)
        encoding = self._infer_output_layout(operand0, operand1)

        with Location.unknown(self.jit_ctx.ctx):
            return RankedTensorType.get(shape, element_type, encoding)

    def create_operation(self, *args, **kwargs):
        """Create binary operation."""
        if len(args) < 2:
            raise ValueError(f"{self.op_name} requires at least 2 arguments")

        operands = self._get_operands(args)
        operand0, operand1 = operands[0], operands[1]

        # Check if either operand is a scalar
        operand0_is_scalar = isinstance(operand0, (int, float, bool))
        operand1_is_scalar = isinstance(operand1, (int, float, bool))

        if operand0_is_scalar and operand1_is_scalar:
            raise ValueError(f"{self.op_name} requires at least one tensor operand")

        # Get the tensor operand for type inference and reference
        tensor_operand = operand1 if operand0_is_scalar else operand0

        # Convert scalars to MLIR constant tensors
        if operand0_is_scalar:
            operand0 = self._create_scalar_tensor_constant(operand0, tensor_operand)
        if operand1_is_scalar:
            operand1 = self._create_scalar_tensor_constant(operand1, tensor_operand)

        # Infer result type from the tensor operand
        result_type = self._infer_result_type(tensor_operand, tensor_operand)

        # Get the TTIR operation constructor (handles name mapping like divide -> div)
        op_constructor = getattr(ttir, get_ttir_name(self.op_name))

        with InsertionPoint(self.jit_ctx.func_bb), Location.unknown(self.jit_ctx.ctx):
            op_result = op_constructor(result=result_type, lhs=operand0, rhs=operand1)

        return self._finalize_result(op_result, args)


class ReductionOpHandler(BaseOpHandler):
    """Handler for reduction operations: sum, mean, max, min."""

    def __init__(self, jit_ctx, op_name):
        super().__init__(jit_ctx)
        self.op_name = op_name
        if op_name not in reduction_ops:
            raise ValueError(f"Unknown reduction operation: {op_name}")

    def _infer_output_shape(self, operand_type, dim_arg, keep_dim):
        """
        Infer output shape based on reduction parameters.

        Cases:
        1. dim specified + keepdim=True: dimension becomes 1
        2. dim specified + keepdim=False: dimension is removed
        3. dim=None (full reduction) + keepdim=True: all dimensions become 1
        4. dim=None (full reduction) + keepdim=False: scalar result (empty shape [])

        Returns:
            List representing the output shape (can be empty for scalar results).
        """
        original_shape = list(operand_type.shape)

        if dim_arg is None or len(dim_arg) == 0:
            # Full reduction over all dimensions
            if keep_dim:
                # All dimensions become 1
                return [1] * len(original_shape)
            else:
                # Scalar result - empty shape
                return []

        # Partial reduction over specified dimensions
        if keep_dim:
            # Keep the dimension but set it to 1
            new_shape = original_shape.copy()
            for dim in dim_arg:
                if 0 <= dim < len(new_shape):
                    new_shape[dim] = 1
            return new_shape
        else:
            # Remove the dimension(s)
            new_shape = [s for i, s in enumerate(original_shape) if i not in dim_arg]
            return new_shape  # Can be empty if all dims removed

    def _infer_result_type(self, operand, dim_arg, keep_dim):
        """Infer result type based on reduction parameters with default layout."""
        operand_type = operand.type
        element_type = operand_type.element_type

        new_shape = self._infer_output_shape(operand_type, dim_arg, keep_dim)
        encoding = self._default_dram_interleaved_layout(element_type, new_shape)

        with Location.unknown(self.jit_ctx.ctx):
            # For scalar results, new_shape is [] which creates a 0D tensor
            return RankedTensorType.get(new_shape, element_type, encoding)

    def create_operation(self, *args, **kwargs):
        """Create reduction operation."""
        if len(args) < 1:
            raise ValueError(f"{self.op_name} requires at least 1 argument")

        operands = self._get_operands(args)
        operand = operands[0]

        # Extract kwargs
        dim_arg = kwargs.get("dim", None)
        if dim_arg is not None:
            dim_arg = [dim_arg]
        keep_dim = kwargs.get("keepdim", False)

        # Infer result type
        result_type = self._infer_result_type(operand, dim_arg, keep_dim)

        # Get the TTIR operation constructor
        op_constructor = getattr(ttir, get_ttir_name(self.op_name))

        with InsertionPoint(self.jit_ctx.func_bb), Location.unknown(self.jit_ctx.ctx):
            op_result = op_constructor(
                result=result_type, input=operand, keep_dim=keep_dim, dim_arg=dim_arg
            )

        return self._finalize_result(op_result, args)


class MatmulOpHandler(BaseOpHandler):
    """Handler for matrix multiplication operation."""

    def _infer_result_type(self, lhs_type, rhs_type):
        """Infer result type for matmul using create_output_tensor."""
        # Use create_output_tensor which handles matmul output layout inference
        return create_output_tensor(
            self.jit_ctx.ctx, "matmul", [lhs_type, rhs_type], CREATE_INTERMEDIATE_LAYOUT
        )

    def create_operation(self, *args, **kwargs):
        """Create matmul operation."""
        if len(args) < 2:
            raise ValueError("matmul requires at least 2 arguments")

        operands = self._get_operands(args)
        lhs, rhs = operands[0], operands[1]

        # Extract kwargs
        transpose_a = kwargs.get("transpose_a", False)
        transpose_b = kwargs.get("transpose_b", False)

        # Infer result type using create_output_tensor
        result_type = self._infer_result_type(lhs.type, rhs.type)

        # Create the operation
        with InsertionPoint(self.jit_ctx.func_bb), Location.unknown(self.jit_ctx.ctx):
            op_result = ttir.matmul(
                result=result_type,
                a=lhs,
                b=rhs,
                transpose_a=transpose_a,
                transpose_b=transpose_b,
            )

        return self._finalize_result(op_result, args)


class PermuteOpHandler(BaseOpHandler):
    """Handler for permute operation."""

    def _apply_permutation(self, shape, permutation):
        """
        Apply permutation to shape to get output shape.

        For permute, output_shape[i] = input_shape[permutation[i]]
        Example: input shape [2, 3, 4], permutation [2, 0, 1] -> output shape [4, 2, 3]
        """
        return [shape[p] for p in permutation]

    def _infer_result_type(self, operand, permutation):
        """Infer result type based on permutation."""
        operand_type = operand.type
        element_type = operand_type.element_type
        input_shape = list(operand_type.shape)

        output_shape = self._apply_permutation(input_shape, permutation)
        encoding = self._default_dram_interleaved_layout(element_type, output_shape)

        with Location.unknown(self.jit_ctx.ctx):
            return RankedTensorType.get(output_shape, element_type, encoding)

    def create_operation(self, *args, **kwargs):
        """Create permute operation."""
        if len(args) < 1:
            raise ValueError("permute requires at least 1 argument (input tensor)")

        operands = self._get_operands(args)
        operand = operands[0]

        # Extract permutation from kwargs
        permutation = kwargs.get("permutation", None)
        if permutation is None:
            raise ValueError("permute requires 'permutation' keyword argument")

        # Validate permutation - only check what's needed to prevent Python crash.
        # Semantic validation (length, uniqueness) is handled by MLIR verify().
        input_rank = len(operand.type.shape)
        if any(p >= input_rank for p in permutation):
            raise ValueError(
                f"Permutation contains out-of-bounds index for tensor with rank {input_rank}"
            )

        # Infer result type
        result_type = self._infer_result_type(operand, permutation)

        # Create the operation
        with InsertionPoint(self.jit_ctx.func_bb), Location.unknown(self.jit_ctx.ctx):
            op_result = ttir.permute(
                result=result_type,
                input=operand,
                permutation=permutation,
            )

        return self._finalize_result(op_result, args)


class TransposeOpHandler(BaseOpHandler):
    """Handler for transpose operation - swaps two dimensions of a tensor."""

    def _swap_dimensions(self, shape, dim0, dim1):
        """
        Swap two dimensions in shape to get output shape.

        For transpose, output_shape is the same as input_shape except
        dimensions at dim0 and dim1 are swapped.
        Example: input shape [2, 3, 4], dim0=0, dim1=2 -> output shape [4, 3, 2]
        """

        output_shape = list(shape)
        output_shape[dim0], output_shape[dim1] = output_shape[dim1], output_shape[dim0]
        return output_shape

    def _infer_result_type(self, operand, dim0, dim1):
        """Infer result type based on dimensions to swap."""
        operand_type = operand.type
        element_type = operand_type.element_type
        input_shape = list(operand_type.shape)
        rank = len(input_shape)

        dim0 = self._normalize_dim(dim0, rank)
        dim1 = self._normalize_dim(dim1, rank)

        output_shape = self._swap_dimensions(input_shape, dim0, dim1)
        encoding = self._default_dram_interleaved_layout(element_type, output_shape)

        with Location.unknown(self.jit_ctx.ctx):
            return RankedTensorType.get(output_shape, element_type, encoding)

    def create_operation(self, *args, **kwargs):
        """Create transpose operation."""
        if len(args) < 1:
            raise ValueError("transpose requires at least 1 argument (input tensor)")

        operands = self._get_operands(args)
        operand = operands[0]

        # Extract dim0 and dim1 from kwargs (default to swapping first two dims)
        dim0 = kwargs.get("dim0", 0)
        dim1 = kwargs.get("dim1", 1)

        # Validate dimensions - needed to prevent Python IndexError in _swap_dimensions.
        # Semantic validation is also done by MLIR verify().
        input_rank = len(operand.type.shape)
        norm_dim0 = self._normalize_dim(dim0, input_rank)
        norm_dim1 = self._normalize_dim(dim1, input_rank)

        # Infer result type
        result_type = self._infer_result_type(operand, norm_dim0, norm_dim1)

        # Create the operation
        with InsertionPoint(self.jit_ctx.func_bb), Location.unknown(self.jit_ctx.ctx):
            op_result = ttir.transpose(
                result=result_type,
                input=operand,
                dim0=dim0,
                dim1=dim1,
            )

        return self._finalize_result(op_result, args)


class ReshapeOpHandler(BaseOpHandler):
    """Handler for reshape operation - changes tensor shape while preserving element count."""

    def _resolve_reshape(self, input_shape, new_shape):
        """
        Resolve -1 in new_shape if present.

        This is needed because MLIR tensor types require concrete dimensions.
        Element count validation is delegated to MLIR verify().

        Returns the resolved new_shape with -1 replaced if present.
        """
        # Check for -1 in new_shape (infer dimension)
        infer_idx = None
        for i, dim in enumerate(new_shape):
            if dim == -1:
                if infer_idx is not None:
                    raise ValueError("Only one dimension can be -1 in reshape")
                infer_idx = i
            elif dim <= 0:
                raise ValueError(f"Invalid dimension {dim} in reshape shape")

        if infer_idx is not None:
            # Infer the -1 dimension - needed to build result type
            input_elements = 1
            for dim in input_shape:
                input_elements *= dim

            known_elements = 1
            for i, dim in enumerate(new_shape):
                if i != infer_idx:
                    known_elements *= dim

            if input_elements % known_elements != 0:
                raise ValueError(
                    f"Cannot infer dimension: {input_elements} elements "
                    f"not divisible by {known_elements}"
                )
            inferred_dim = input_elements // known_elements
            new_shape = list(new_shape)
            new_shape[infer_idx] = inferred_dim

        # Element count validation is handled by ReshapeOp::verify()
        return list(new_shape)

    def _infer_result_type(self, operand, new_shape):
        """Infer result type based on new shape."""
        operand_type = operand.type
        element_type = operand_type.element_type
        input_shape = list(operand_type.shape)

        output_shape = self._resolve_reshape(input_shape, new_shape)
        encoding = self._default_dram_interleaved_layout(element_type, output_shape)

        with Location.unknown(self.jit_ctx.ctx):
            return RankedTensorType.get(output_shape, element_type, encoding)

    def create_operation(self, *args, **kwargs):
        """Create reshape operation."""
        if len(args) < 1:
            raise ValueError("reshape requires at least 1 argument (input tensor)")

        operands = self._get_operands(args)
        operand = operands[0]

        # Extract shape from kwargs or second positional arg
        new_shape = kwargs.get("shape", None)
        if new_shape is None and len(args) >= 2:
            new_shape = args[1]
        if new_shape is None:
            raise ValueError("reshape requires 'shape' argument")

        # Convert to list if tuple
        new_shape = list(new_shape)

        # Infer result type (validates reshape)
        result_type = self._infer_result_type(operand, new_shape)

        # Resolve -1 dimensions for the attribute
        input_shape = list(operand.type.shape)
        resolved_shape = self._resolve_reshape(input_shape, new_shape)

        # Create the operation
        with InsertionPoint(self.jit_ctx.func_bb), Location.unknown(self.jit_ctx.ctx):
            op_result = ttir.reshape(
                result=result_type,
                input=operand,
                shape=resolved_shape,
            )

        return self._finalize_result(op_result, args)


class RearrangeOpHandler(BaseOpHandler):
    """Handler for rearrange operation - uses einops-style pattern to rearrange tensor dimensions."""

    def _parse_dims(self, pattern):
        """
        Parse dimension names from a pattern string.

        Returns a list of lists, where each inner list represents a group of dimensions.
        Single dimensions are in groups of size 1, merged dimensions (in parentheses)
        are in groups of size > 1.

        Examples:
        - "b h w c" -> [["b"], ["h"], ["w"], ["c"]]
        - "b c (h w)" -> [["b"], ["c"], ["h", "w"]]
        - "(b h) w c" -> [["b", "h"], ["w"], ["c"]]
        """
        result = []
        current_group = []
        in_parens = False
        i = 0

        while i < len(pattern):
            c = pattern[i]

            if c == "(":
                in_parens = True
                i += 1
                continue

            if c == ")":
                if current_group:
                    result.append(current_group)
                    current_group = []
                in_parens = False
                i += 1
                continue

            if c == " " and not in_parens and current_group:
                result.append(current_group)
                current_group = []
                i += 1
                continue

            if c == " ":
                i += 1
                continue

            # Parse dimension name
            start = i
            while i < len(pattern) and pattern[i] not in " ()":
                i += 1

            if i > start:
                current_group.append(pattern[start:i])

        if current_group:
            result.append(current_group)

        return result

    def _compute_output_shape(self, input_shape, pattern):
        """
        Compute output shape based on einops-style pattern.

        The pattern is of the form "input_pattern -> output_pattern" where:
        - Input pattern names the input dimensions
        - Output pattern describes how to rearrange them

        For merging (parentheses in output): dimensions are multiplied
        For reordering: dimensions are permuted

        Note: Splitting dimensions (parentheses in input) is currently unsupported.
        """
        # Split pattern into input and output parts
        parts = pattern.split("->")
        if len(parts) != 2:
            raise ValueError("Pattern must contain exactly one '->' separator")

        input_pattern = parts[0].strip()
        output_pattern = parts[1].strip()

        # Parse input dims to build dimension name -> size mapping
        input_dims = self._parse_dims(input_pattern)

        # Build map from dimension name to its size
        dim_to_size = {}
        for pos, group in enumerate(input_dims):
            if len(group) > 1:
                # Splitting (parentheses in input) is unsupported
                raise ValueError(
                    "Splitting dimensions (parentheses in input pattern) is not supported #6339"
                )
            if pos >= len(input_shape):
                raise ValueError(
                    f"Pattern has more input dimensions than tensor rank "
                    f"(pattern has {len(input_dims)}, tensor has {len(input_shape)})"
                )
            dim_name = group[0]
            dim_to_size[dim_name] = input_shape[pos]

        # Parse output dims and compute output shape
        output_dims = self._parse_dims(output_pattern)
        output_shape = []

        for group in output_dims:
            # Compute size for this output dimension (multiply if merged)
            dim_size = 1
            for dim_name in group:
                if dim_name not in dim_to_size:
                    raise ValueError(
                        f"Unknown dimension '{dim_name}' in output pattern"
                    )
                dim_size *= dim_to_size[dim_name]
            output_shape.append(dim_size)

        return output_shape

    def _infer_result_type(self, operand, pattern):
        """Infer result type based on pattern."""
        operand_type = operand.type
        element_type = operand_type.element_type
        input_shape = list(operand_type.shape)

        output_shape = self._compute_output_shape(input_shape, pattern)
        encoding = self._default_dram_interleaved_layout(element_type, output_shape)

        with Location.unknown(self.jit_ctx.ctx):
            return RankedTensorType.get(output_shape, element_type, encoding)

    def create_operation(self, *args, **kwargs):
        """Create rearrange operation."""
        if len(args) < 1:
            raise ValueError("rearrange requires at least 1 argument (input tensor)")

        operands = self._get_operands(args)
        operand = operands[0]

        # Extract pattern from kwargs
        pattern = kwargs.get("pattern", None)
        if pattern is None:
            raise ValueError("rearrange requires 'pattern' keyword argument")

        # Infer result type (validates pattern)
        result_type = self._infer_result_type(operand, pattern)

        # Create the operation - pattern must be a StringAttr
        with InsertionPoint(self.jit_ctx.func_bb), Location.unknown(self.jit_ctx.ctx):
            pattern_attr = StringAttr.get(pattern)
            op_result = ttir.rearrange(
                result=result_type,
                input=operand,
                pattern=pattern_attr,
            )

        return self._finalize_result(op_result, args)


class ConcatOpHandler(BaseOpHandler):
    """Handler for concat operation - concatenates tensors along a specified dimension."""

    def _compute_output_shape(self, operand_types, dim):
        """
        Compute output shape for concatenation.

        For concat, the output shape is the same as the input shapes except
        for the concatenation dimension, which is the sum of all input dimensions.
        """
        if not operand_types:
            raise ValueError("concat requires at least one input tensor")

        first_shape = list(operand_types[0].shape)
        rank = len(first_shape)
        normalized_dim = self._normalize_dim(dim, rank)
        # Sum dimensions along concat axis
        concat_dim_size = sum(t.shape[normalized_dim] for t in operand_types)

        # Build output shape
        output_shape = list(first_shape)
        output_shape[normalized_dim] = concat_dim_size

        return output_shape

    def _infer_result_type(self, operands, dim):
        """Infer result type based on input operands and concat dimension."""
        operand_types = [op.type for op in operands]
        element_type = operand_types[0].element_type

        output_shape = self._compute_output_shape(operand_types, dim)
        encoding = self._default_dram_interleaved_layout(element_type, output_shape)

        with Location.unknown(self.jit_ctx.ctx):
            return RankedTensorType.get(output_shape, element_type, encoding)

    def create_operation(self, *args, **kwargs):
        """Create concat operation."""
        if len(args) < 1:
            raise ValueError("concat requires at least 1 input tensor")

        # ttnn.concat takes a list of tensors as first argument: ttnn.concat([a, b], dim=0)
        # Handle both cases: list as first arg, or individual tensors as args
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            input_tensors = args[0]
        else:
            input_tensors = args

        if len(input_tensors) < 1:
            raise ValueError("concat requires at least 1 input tensor")

        operands = self._get_operands(input_tensors)

        # Extract dim from kwargs (default to 0)
        dim = kwargs.get("dim", 0)

        # Infer result type
        result_type = self._infer_result_type(operands, dim)

        # Create the operation
        with InsertionPoint(self.jit_ctx.func_bb), Location.unknown(self.jit_ctx.ctx):
            op_result = ttir.concat(
                result=result_type,
                inputs=operands,
                dim=dim,
            )

        return self._finalize_result(op_result, input_tensors)


class RepeatOpHandler(BaseOpHandler):
    """Handler for repeat operation - repeats tensor along each dimension."""

    def _compute_output_shape(self, operand_type, repeat_dimensions):
        """
        Compute output shape for repeat operation.

        Output shape is input_shape[i] * repeat_dimensions[i] for each dimension.
        """
        input_shape = list(operand_type.shape)
        input_rank = len(input_shape)

        # Fundamental verification: repeat_dimensions must match input rank
        if len(repeat_dimensions) != input_rank:
            raise ValueError(
                f"repeat_dimensions length {len(repeat_dimensions)} doesn't match "
                f"input tensor rank {input_rank}"
            )

        # Fundamental verification: all repeat values must be positive
        for i, rep in enumerate(repeat_dimensions):
            if rep <= 0:
                raise ValueError(
                    f"Repeat dimension at index {i} must be greater than 0, got {rep}"
                )

        # Compute output shape: input_shape[i] * repeat_dimensions[i]
        output_shape = [
            input_shape[i] * repeat_dimensions[i] for i in range(input_rank)
        ]
        return output_shape

    def _infer_result_type(self, operand, repeat_dimensions):
        """Infer result type based on input operand and repeat dimensions."""
        operand_type = operand.type
        element_type = operand_type.element_type

        output_shape = self._compute_output_shape(operand_type, repeat_dimensions)
        encoding = self._default_dram_interleaved_layout(element_type, output_shape)

        with Location.unknown(self.jit_ctx.ctx):
            return RankedTensorType.get(output_shape, element_type, encoding)

    def create_operation(self, *args, **kwargs):
        """Create repeat operation."""
        # Signature: ttnn.repeat(tensor, repeat_dims) where repeat_dims is a list or ttnn.Shape
        if len(args) < 2:
            raise ValueError(
                "repeat requires 2 arguments: (input_tensor, repeat_dimensions)"
            )

        operands = self._get_operands(args[:1])
        operand = operands[0]

        # Second argument is the repeat_dimensions (list or ttnn.Shape)
        repeat_dimensions = args[1]

        # Convert to list if needed (handles ttnn.Shape, numpy arrays, tuples, etc.)
        if hasattr(repeat_dimensions, "tolist"):
            repeat_dimensions = repeat_dimensions.tolist()
        elif hasattr(repeat_dimensions, "__iter__") and not isinstance(
            repeat_dimensions, (list, tuple)
        ):
            # Handle ttnn.Shape or similar iterable objects
            repeat_dimensions = list(repeat_dimensions)
        else:
            repeat_dimensions = list(repeat_dimensions)

        # Infer result type
        result_type = self._infer_result_type(operand, repeat_dimensions)

        # Create the operation
        with InsertionPoint(self.jit_ctx.func_bb), Location.unknown(self.jit_ctx.ctx):
            op_result = ttir.repeat(
                result=result_type,
                input=operand,
                repeat_dimensions=repeat_dimensions,
            )

        return self._finalize_result(op_result, args)


class GatherOpHandler(BaseOpHandler):
    """Handler for gather operation - gathers elements from input along a dimension.

    The gather operation follows torch.gather semantics:
    - input: source tensor with shape [d0, d1, ..., dn]
    - dim: the dimension to gather along
    - index: tensor with indices, must have same rank as input
    - output: same shape as index tensor

    For example, torch.gather(input, dim=1, index):
      output[i][j][k] = input[i][index[i][j][k]][k]

    This is converted to TTIR's StableHLO-style gather operation.
    """

    def _compute_output_shape(self, index_type):
        """
        Compute output shape for gather operation.

        For torch.gather, the output shape is the same as the index tensor shape.
        """
        return list(index_type.shape)

    def _compute_gather_attributes(self, input_type, index_type, dim):
        """
        Compute StableHLO-style gather attributes from torch.gather parameters.

        For torch.gather(input, dim, index):
        - Each element in output corresponds to one element from input
        - index specifies which element to take along 'dim'
        - Other dimensions are matched between input, index, and output

        StableHLO gather attributes for torch.gather:
        - offset_dims: dimensions of output that come from the slice (empty for element gather)
        - collapsed_slice_dims: all dimensions since we take single elements
        - start_index_map: [dim] - the dimension being indexed
        - index_vector_dim: rank of index (indices are scalars)
        - slice_sizes: all ones (gathering individual elements)
        """
        input_rank = len(input_type.shape)
        index_rank = len(index_type.shape)

        # Fundamental verification: dim must be valid
        if dim < 0 or dim >= input_rank:
            raise ValueError(
                f"Dimension {dim} out of range for tensor with rank {input_rank}"
            )

        # Fundamental verification: index rank must match input rank
        if index_rank != input_rank:
            raise ValueError(
                f"Index tensor rank ({index_rank}) must match input tensor rank ({input_rank})"
            )

        # For torch.gather, we gather individual elements
        # offset_dims: empty - output shape comes entirely from index tensor batch dims
        offset_dims = []

        # collapsed_slice_dims: all dimensions since we take size-1 slices
        collapsed_slice_dims = list(range(input_rank))

        # operand_batching_dims and start_indices_batching_dims: empty
        operand_batching_dims = []
        start_indices_batching_dims = []

        # start_index_map: the dimension we're gathering along
        start_index_map = [dim]

        # index_vector_dim: equal to index rank (indices are scalars, not vectors)
        index_vector_dim = index_rank

        # slice_sizes: all ones (gathering individual elements)
        slice_sizes = [1] * input_rank

        return {
            "offset_dims": offset_dims,
            "collapsed_slice_dims": collapsed_slice_dims,
            "operand_batching_dims": operand_batching_dims,
            "start_indices_batching_dims": start_indices_batching_dims,
            "start_index_map": start_index_map,
            "index_vector_dim": index_vector_dim,
            "slice_sizes": slice_sizes,
            "indices_are_sorted": False,
        }

    def _infer_result_type(self, input_operand, index_operand, dim):
        """Infer result type based on index tensor shape."""
        input_type = input_operand.type
        index_type = index_operand.type
        element_type = input_type.element_type

        # Validate dim is in range
        input_rank = len(input_type.shape)
        _ = self._normalize_dim(dim, input_rank)

        # Output shape is the same as index shape
        output_shape = self._compute_output_shape(index_type)
        encoding = self._default_dram_interleaved_layout(element_type, output_shape)

        with Location.unknown(self.jit_ctx.ctx):
            return RankedTensorType.get(output_shape, element_type, encoding)

    def create_operation(self, *args, **kwargs):
        """Create gather operation."""
        if len(args) < 2:
            raise ValueError("gather requires at least 2 arguments: input and dim")

        operands = self._get_operands(args[:1])
        input_operand = operands[0]

        # Extract dim (second positional argument)
        dim = args[1]
        if not isinstance(dim, int):
            raise ValueError(f"dim must be an integer, got {type(dim)}")

        # Extract index from kwargs
        index = kwargs.get("index", None)
        if index is None:
            raise ValueError("gather requires 'index' keyword argument")

        # Resolve index operand
        index_operand = self._resolve_operand(index, 1, self.jit_ctx)

        # Normalize dim
        input_rank = len(input_operand.type.shape)
        normalized_dim = self._normalize_dim(dim, input_rank)

        # Compute gather attributes
        attrs = self._compute_gather_attributes(
            input_operand.type, index_operand.type, normalized_dim
        )

        # Infer result type
        result_type = self._infer_result_type(
            input_operand, index_operand, normalized_dim
        )

        # Create the operation
        with InsertionPoint(self.jit_ctx.func_bb), Location.unknown(self.jit_ctx.ctx):
            op_result = ttir.gather(
                result=result_type,
                input=input_operand,
                start_indices=index_operand,
                offset_dims=attrs["offset_dims"],
                collapsed_slice_dims=attrs["collapsed_slice_dims"],
                operand_batching_dims=attrs["operand_batching_dims"],
                start_indices_batching_dims=attrs["start_indices_batching_dims"],
                start_index_map=attrs["start_index_map"],
                index_vector_dim=attrs["index_vector_dim"],
                slice_sizes=attrs["slice_sizes"],
                indices_are_sorted=attrs["indices_are_sorted"],
            )

        return self._finalize_result(op_result, args)


class EmbeddingOpHandler(BaseOpHandler):
    """Handler for embedding operation - performs embedding lookup from a table.

    The embedding operation takes an input tensor of indices and a weight tensor
    (embedding table). For each index in the input tensor, it retrieves the
    corresponding row from the weight tensor.

    Semantics:
    - input: tensor of indices (at most 2D, e.g., [batch, seq_len])
    - weight: embedding table (effectively 2D, e.g., [vocab_size, embedding_dim])
    - output: input.shape + [embedding_dim]
    """

    def _compute_output_shape(self, input_type, weight_type):
        """
        Compute output shape for embedding operation.

        Output shape is (*input.shape, embedding_dim) where embedding_dim
        is the last dimension of the weight tensor.
        """
        input_shape = list(input_type.shape)
        weight_shape = list(weight_type.shape)

        # Fundamental verification: input must be at most 2D
        if len(input_shape) > 2:
            raise ValueError(
                f"Embedding input must be at most 2D tensor, got {len(input_shape)}D"
            )

        # Fundamental verification: weight must be at least 2D
        if len(weight_shape) < 2:
            raise ValueError(
                f"Embedding weight must be at least 2D tensor, got {len(weight_shape)}D"
            )

        # Get embedding dimension (last dimension of weight)
        embedding_dim = weight_shape[-1]

        # Output shape is input_shape + [embedding_dim]
        output_shape = input_shape + [embedding_dim]
        return output_shape

    def _infer_result_type(self, input_operand, weight_operand):
        """Infer result type based on input and weight operands."""
        input_type = input_operand.type
        weight_type = weight_operand.type
        # Use weight's element type for output (indices may be integers)
        element_type = weight_type.element_type

        output_shape = self._compute_output_shape(input_type, weight_type)
        encoding = self._default_dram_interleaved_layout(element_type, output_shape)

        with Location.unknown(self.jit_ctx.ctx):
            return RankedTensorType.get(output_shape, element_type, encoding)

    def create_operation(self, *args, **kwargs):
        """Create embedding operation."""
        if len(args) < 2:
            raise ValueError("embedding requires 2 arguments: input and weight")

        operands = self._get_operands(args[:2])
        input_operand = operands[0]
        weight_operand = operands[1]

        # Infer result type
        result_type = self._infer_result_type(input_operand, weight_operand)

        # Create the operation
        with InsertionPoint(self.jit_ctx.func_bb), Location.unknown(self.jit_ctx.ctx):
            op_result = ttir.embedding(
                result=result_type,
                input=input_operand,
                weight=weight_operand,
            )

        return self._finalize_result(op_result, args)


class ClampOpHandler(UnaryOpHandler):
    """Handler for clamp operation.

    Supports two modes:
    1. Scalar bounds: ttnn.clamp(input, min=2.0, max=5.0) -> ttir.clamp_scalar
    2. Tensor bounds: ttnn.clamp(input, min=min_tensor, max=max_tensor) -> ttir.clamp_tensor
    """

    _F32_MAX = 3.4028235e38

    def __init__(self, jit_ctx):
        BaseOpHandler.__init__(self, jit_ctx)
        self.op_name = "clamp"

    def _normalize_scalar_value(self, value):
        """Normalize Python scalar for MLIR attribute/tensor creation."""
        if isinstance(value, float) and math.isinf(value):
            return math.copysign(self._F32_MAX, value)
        return value

    def _convert_scalar_to_attr(self, value, element_type):
        """Convert Python scalar to MLIR attribute matching element type."""
        value = self._normalize_scalar_value(value)

        # Convert to appropriate MLIR attribute
        with Location.unknown(self.jit_ctx.ctx):
            if isinstance(element_type, (F32Type, F64Type, BF16Type)):
                return FloatAttr.get(F32Type.get(self.jit_ctx.ctx), float(value))
            elif isinstance(element_type, IntegerType):
                return IntegerAttr.get(element_type, int(value))
            else:
                raise ValueError(f"Unsupported element type for clamp: {element_type}")

    def _is_scalar(self, value):
        """Check if value is a scalar (not a tensor)."""
        return isinstance(value, (int, float, bool))

    def create_operation(self, *args, **kwargs):
        """Create clamp operation.

        Signature: ttnn.clamp(input, min=..., max=...) or ttnn.clamp(input, min_val, max_val)
        Supports both keyword and positional arguments for flexibility.
        """
        if len(args) < 1:
            raise ValueError("clamp requires at least 1 argument (input tensor)")

        operands = self._get_operands(args[:1])
        input_operand = operands[0]

        # Check if input is a scalar - clamp requires tensor operands
        if isinstance(input_operand, (int, float, bool)):
            raise TypeError("clamp requires a tensor operand, not a scalar")

        # Extract min and max - support both positional and keyword args
        min_val = kwargs.get("min", None)
        max_val = kwargs.get("max", None)

        # If not in kwargs, check positional args
        if min_val is None and len(args) >= 2:
            min_val = args[1]
        if max_val is None and len(args) >= 3:
            max_val = args[2]

        if min_val is None and max_val is None:
            raise ValueError("clamp requires at least one of 'min' or 'max' arguments")

        # Infer result type
        result_type = self._infer_result_type(input_operand)

        # Determine if we need scalar or tensor variant
        min_is_scalar = min_val is None or self._is_scalar(min_val)
        max_is_scalar = max_val is None or self._is_scalar(max_val)

        if min_is_scalar and max_is_scalar:
            # Use ttir.clamp_scalar
            element_type = input_operand.type.element_type

            # Convert to attributes, using very large values if None
            if min_val is None:
                min_val = float("-inf")
            if max_val is None:
                max_val = float("inf")

            min_attr = self._convert_scalar_to_attr(min_val, element_type)
            max_attr = self._convert_scalar_to_attr(max_val, element_type)

            with InsertionPoint(self.jit_ctx.func_bb), Location.unknown(
                self.jit_ctx.ctx
            ):
                op_result = ttir.clamp_scalar(
                    result=result_type,
                    input=input_operand,
                    min=min_attr,
                    max=max_attr,
                )
        else:
            # Use ttir.clamp_tensor
            # Resolve min and max operands
            if min_val is None:
                # Create a tensor filled with -inf
                min_operand = self._create_scalar_tensor_constant(
                    self._normalize_scalar_value(float("-inf")), input_operand
                )
            elif self._is_scalar(min_val):
                min_operand = self._create_scalar_tensor_constant(
                    self._normalize_scalar_value(min_val), input_operand
                )
            else:
                min_operand = self._resolve_operand(min_val, 1, self.jit_ctx)

            if max_val is None:
                # Create a tensor filled with inf
                max_operand = self._create_scalar_tensor_constant(
                    self._normalize_scalar_value(float("inf")), input_operand
                )
            elif self._is_scalar(max_val):
                max_operand = self._create_scalar_tensor_constant(
                    self._normalize_scalar_value(max_val), input_operand
                )
            else:
                max_operand = self._resolve_operand(max_val, 2, self.jit_ctx)

            with InsertionPoint(self.jit_ctx.func_bb), Location.unknown(
                self.jit_ctx.ctx
            ):
                op_result = ttir.clamp_tensor(
                    result=result_type,
                    input=input_operand,
                    min=min_operand,
                    max=max_operand,
                )

        return self._finalize_result(op_result, args)


class TTNNJitNamespaceUpdater:
    """Namespace updater that provides jit functions for tracing mode."""

    def __init__(self, jit_ctx):
        self.jit_ctx = jit_ctx
        self.register_all_operations(jit_ctx)

    def register_all_operations(self, jit_ctx):
        """Register all operations in the namespace."""
        ######################## Unary operations ########################
        self._unary_handlers = {
            op_name: UnaryOpHandler(jit_ctx, op_name) for op_name in unary_ops
        }
        for op_name in unary_ops:
            setattr(
                self,
                op_name,
                partial(self._call_handler, self._unary_handlers[op_name]),
            )

        ######################## Binary operations ########################
        self._binary_handlers = {
            op_name: BinaryOpHandler(jit_ctx, op_name) for op_name in binary_ops
        }
        for op_name in binary_ops:
            setattr(
                self,
                op_name,
                partial(self._call_handler, self._binary_handlers[op_name]),
            )
        # Handles the special case for pow and div:
        for internal_name, ttir_name in TTIR_NAME_MAP.items():
            if internal_name in binary_ops and ttir_name != internal_name:
                setattr(
                    self,
                    ttir_name,
                    partial(self._call_handler, self._binary_handlers[internal_name]),
                )

        ######################## Reduction operations ########################
        self._reduction_handlers = {
            op_name: ReductionOpHandler(jit_ctx, op_name) for op_name in reduction_ops
        }
        for op_name in reduction_ops:
            setattr(
                self,
                op_name,
                partial(self._call_handler, self._reduction_handlers[op_name]),
            )

        ######################## Matmul operation ########################
        self._matmul_handler = MatmulOpHandler(jit_ctx)
        self.matmul = partial(self._call_handler, self._matmul_handler)

        ######################## TM (Tensor Manipulation) operations ########################
        self._permute_handler = PermuteOpHandler(jit_ctx)
        self.permute = partial(self._call_handler, self._permute_handler)

        self._transpose_handler = TransposeOpHandler(jit_ctx)
        self.transpose = partial(self._call_handler, self._transpose_handler)

        self._reshape_handler = ReshapeOpHandler(jit_ctx)
        self.reshape = partial(self._call_handler, self._reshape_handler)

        self._rearrange_handler = RearrangeOpHandler(jit_ctx)
        self.rearrange = partial(self._call_handler, self._rearrange_handler)

        ######################## Concat operation ########################
        self._concat_handler = ConcatOpHandler(jit_ctx)
        self.concat = partial(self._call_handler, self._concat_handler)

        ######################## Repeat operation ########################
        self._repeat_handler = RepeatOpHandler(jit_ctx)
        self.repeat = partial(self._call_handler, self._repeat_handler)

        ######################## Embedding operation ########################
        self._embedding_handler = EmbeddingOpHandler(jit_ctx)
        self.embedding = partial(self._call_handler, self._embedding_handler)

        ######################## Gather operation ########################
        self._gather_handler = GatherOpHandler(jit_ctx)
        self.gather = partial(self._call_handler, self._gather_handler)

        ######################## Clamp operation ########################
        self._clamp_handler = ClampOpHandler(jit_ctx)
        self.clamp = partial(self._call_handler, self._clamp_handler)

    def _call_handler(self, handler, *args, **kwargs):
        """Call the handler's create_operation method."""
        return handler.create_operation(*args, **kwargs)
