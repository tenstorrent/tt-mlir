# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from functools import partial

from ttmlir.dialects import ttir
from ttmlir.ir import InsertionPoint, Location, RankedTensorType

from ttnn_jit._src.operand_resolver import OperandResolver


class ResultWrapper:
    """Wrapper to track MLIR operation results."""

    def __init__(self, mlir_value, original_value=None):
        self.mlir_value = mlir_value
        self.original_value = original_value


class BaseOpHandler(ABC):
    """Abstract base class for all operation handlers."""

    def __init__(self, jit_ctx):
        self.jit_ctx = jit_ctx
        self.resolver = OperandResolver()

    def _get_operands(self, args):
        """Get MLIR operands from Python arguments."""
        operands = []
        for i, arg in enumerate(args):
            if arg is None:
                raise ValueError(f"Argument at index {i} is None")
            operand = self.resolver.resolve_operand(arg, i, self.jit_ctx)
            operands.append(operand)
        return operands

    def _create_result_wrapper(self, mlir_value, original_value=None):
        """Create a ResultWrapper for the operation result."""
        return ResultWrapper(mlir_value, original_value)

    def _store_result(self, arg_id, mlir_value):
        """Store result in value_map if not an original function argument."""
        if arg_id not in self.jit_ctx.func_arg_ids:
            self.jit_ctx.value_map[arg_id] = mlir_value

    @abstractmethod
    def create_operation(self, *args, **kwargs):
        """Create the MLIR operation. Must be implemented by subclasses."""
        pass


class BinaryOpHandler(BaseOpHandler):
    """Handler for binary operations: add, subtract, multiply."""

    def __init__(self, jit_ctx, op_name):
        super().__init__(jit_ctx)
        self.op_name = op_name
        self.op_map = {
            "add": ttir.add,
            "subtract": ttir.subtract,
            "multiply": ttir.multiply,
        }

    def _infer_result_type(self, operand0, operand1):
        """Infer result type from operands (no layout for intermediate results)."""
        # Use the first operand's element type
        element_type = operand0.type.element_type

        # For intermediate results, don't preserve encoding (layout)
        # Use shape from first operand
        shape = list(operand0.type.shape)

        with Location.unknown(self.jit_ctx.ctx):
            return RankedTensorType.get(shape, element_type, None)

    def create_operation(self, *args, **kwargs):
        """Create binary operation (add, subtract, or multiply)."""
        if len(args) < 2:
            raise ValueError(f"{self.op_name} requires at least 2 arguments")

        operands = self._get_operands(args)
        operand0, operand1 = operands[0], operands[1]

        # Infer result type (no layout for intermediate results)
        result_type = self._infer_result_type(operand0, operand1)

        # Get the operation constructor
        op_constructor = self.op_map[self.op_name]

        # Create the operation
        with InsertionPoint(self.jit_ctx.func_bb), Location.unknown(self.jit_ctx.ctx):
            if self.op_name == "add":
                op_result = op_constructor(
                    result=result_type, lhs=operand0, rhs=operand1
                )
            elif self.op_name == "subtract":
                op_result = op_constructor(
                    result=result_type, lhs=operand0, rhs=operand1
                )
            elif self.op_name == "multiply":
                op_result = op_constructor(
                    result=result_type, lhs=operand0, rhs=operand1
                )
            else:
                raise ValueError(f"Unknown binary operation: {self.op_name}")

        # Store result if applicable
        if args:
            arg_id = id(args[0])
            self._store_result(arg_id, op_result)

        return self._create_result_wrapper(op_result, args[0] if args else None)


class ReductionOpHandler(BaseOpHandler):
    """Handler for reduction operations: sum."""

    def _infer_result_type(self, operand, dim_arg, keep_dim):
        """Infer result type based on reduction parameters."""
        operand_type = operand.type
        element_type = operand_type.element_type

        # For intermediate results, don't preserve encoding (layout)
        encoding = None

        with Location.unknown(self.jit_ctx.ctx):
            if dim_arg and len(dim_arg) > 0:
                original_shape = list(operand_type.shape)
                if keep_dim:
                    # Keep the dimension but set it to 1
                    new_shape = original_shape.copy()
                    for dim in dim_arg:
                        if 0 <= dim < len(new_shape):
                            new_shape[dim] = 1
                    result_type = RankedTensorType.get(
                        new_shape, element_type, encoding
                    )
                else:
                    # Remove the dimension(s)
                    new_shape = [
                        s for i, s in enumerate(original_shape) if i not in dim_arg
                    ]
                    result_type = (
                        RankedTensorType.get(new_shape, element_type, encoding)
                        if new_shape
                        else element_type
                    )
            else:
                # No dim specified, use operand type (but without encoding)
                shape = list(operand_type.shape)
                result_type = RankedTensorType.get(shape, element_type, encoding)

        return result_type

    def create_operation(self, *args, **kwargs):
        """Create sum operation."""
        if len(args) < 1:
            raise ValueError("sum requires at least 1 argument")

        operands = self._get_operands(args)
        operand = operands[0]

        # Extract kwargs
        dim_arg = kwargs.get("dim", None)
        if dim_arg is not None:
            dim_arg = [dim_arg]
        keep_dim = kwargs.get("keepdim", False)

        # Infer result type
        result_type = self._infer_result_type(operand, dim_arg, keep_dim)

        # Create the operation
        with InsertionPoint(self.jit_ctx.func_bb), Location.unknown(self.jit_ctx.ctx):
            op_result = ttir.sum(
                result=result_type, input=operand, keep_dim=keep_dim, dim_arg=dim_arg
            )

        # Store result if applicable
        if args:
            arg_id = id(args[0])
            self._store_result(arg_id, op_result)

        return self._create_result_wrapper(op_result, args[0] if args else None)


class MatmulOpHandler(BaseOpHandler):
    """Handler for matrix multiplication operation."""

    def _infer_output_shape(self, lhs_shape, rhs_shape):
        """
        Infer output shape for matmul: m×k × k×n = m×n.

        For batched matmul: [..., m, k] × [..., k, n] = [..., m, n]
        """
        # Handle batched dimensions
        if len(lhs_shape) > 2 and len(rhs_shape) > 2:
            # Check if batch dimensions match
            lhs_batch = lhs_shape[:-2]
            rhs_batch = rhs_shape[:-2]
            if lhs_batch != rhs_batch:
                raise ValueError(
                    f"Batch dimensions must match: {lhs_batch} != {rhs_batch}"
                )
            batch_dims = lhs_batch
            m, k = lhs_shape[-2:]
            k_rhs, n = rhs_shape[-2:]
            if k != k_rhs:
                raise ValueError(f"Inner dimensions must match: {k} != {k_rhs}")
            return list(batch_dims) + [m, n]
        elif len(lhs_shape) == 2 and len(rhs_shape) == 2:
            # Simple 2D matmul
            m, k = lhs_shape
            k_rhs, n = rhs_shape
            if k != k_rhs:
                raise ValueError(f"Inner dimensions must match: {k} != {k_rhs}")
            return [m, n]
        else:
            raise ValueError(f"Unsupported matmul shapes: {lhs_shape} × {rhs_shape}")

    def _infer_result_type(self, lhs_type, rhs_type):
        """Infer result type for matmul (no layout for intermediate results)."""
        lhs_shape = list(lhs_type.shape)
        rhs_shape = list(rhs_type.shape)

        # Infer output shape
        output_shape = self._infer_output_shape(lhs_shape, rhs_shape)

        # Use element type from lhs (should match rhs)
        element_type = lhs_type.element_type

        # No layout for intermediate results
        encoding = None

        with Location.unknown(self.jit_ctx.ctx):
            return RankedTensorType.get(output_shape, element_type, encoding)

    def create_operation(self, *args, **kwargs):
        """Create matmul operation."""
        if len(args) < 2:
            raise ValueError("matmul requires at least 2 arguments")

        operands = self._get_operands(args)
        lhs, rhs = operands[0], operands[1]

        # Extract kwargs
        transpose_a = kwargs.get("transpose_a", False)
        transpose_b = kwargs.get("transpose_b", False)

        # Infer result type
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

        # Store result if applicable
        if args:
            arg_id = id(args[0])
            self._store_result(arg_id, op_result)

        return self._create_result_wrapper(op_result, args[0] if args else None)


class TTNNJitNamespaceUpdater:
    """Namespace updater that provides jit functions for tracing mode."""

    def __init__(self, jit_ctx):
        self.jit_ctx = jit_ctx

        # Create handlers
        self._add_handler = BinaryOpHandler(jit_ctx, "add")
        self._subtract_handler = BinaryOpHandler(jit_ctx, "subtract")
        self._multiply_handler = BinaryOpHandler(jit_ctx, "multiply")
        self._sum_handler = ReductionOpHandler(jit_ctx)
        self._matmul_handler = MatmulOpHandler(jit_ctx)

        # Create bound methods
        self.add = partial(self._call_handler, self._add_handler)
        self.subtract = partial(self._call_handler, self._subtract_handler)
        self.multiply = partial(self._call_handler, self._multiply_handler)
        self.sum = partial(self._call_handler, self._sum_handler)
        self.matmul = partial(self._call_handler, self._matmul_handler)

    def _call_handler(self, handler, *args, **kwargs):
        """Call the handler's create_operation method."""
        return handler.create_operation(*args, **kwargs)
