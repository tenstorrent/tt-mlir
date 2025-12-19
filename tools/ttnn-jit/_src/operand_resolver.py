# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Operand resolution logic for tracing-based TTIR generation.

This module provides robust operand resolution that distinguishes between:
- Input function arguments (mapped to MLIR block arguments)
- Intermediate results (from previous operations)
"""


class OperandResolver:
    """
    Handles resolution of Python arguments to MLIR operands.

    Distinguishes between:
    1. Input function arguments -> MLIR block arguments
    2. Intermediate results -> MLIR values from value_map or ResultWrapper
    """

    @staticmethod
    def resolve_operand(arg, arg_index, jit_ctx):
        """
        Resolve a Python argument to its corresponding MLIR operand.

        Resolution order:
        1. Check if arg is a ResultWrapper (has mlir_value attribute)
        2. Check if arg is in value_map (tracked intermediate result)
        3. Fallback to function block argument (input argument)

        Args:
            arg: Python argument (tensor or ResultWrapper)
            arg_index: Index of the argument in the function signature
            jit_ctx: JIT context containing value_map, func_bb, etc.

        Returns:
            MLIR Value corresponding to the argument
        """
        if arg is None:
            raise ValueError(f"Argument at index {arg_index} is None")

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

    @staticmethod
    def is_intermediate_result(arg, jit_ctx):
        """
        Check if an argument is an intermediate result (not an input argument).

        Args:
            arg: Python argument to check
            jit_ctx: JIT context

        Returns:
            True if arg is an intermediate result, False otherwise
        """
        if arg is None:
            return False

        # Check if it's a ResultWrapper
        if hasattr(arg, "mlir_value"):
            return True

        # Check if it's in value_map but not in func_arg_ids
        arg_id = id(arg)
        return arg_id in jit_ctx.value_map and arg_id not in jit_ctx.func_arg_ids

    @staticmethod
    def is_input_argument(arg, jit_ctx):
        """
        Check if an argument is an input function argument.

        Args:
            arg: Python argument to check
            jit_ctx: JIT context

        Returns:
            True if arg is an input argument, False otherwise
        """
        if arg is None:
            return False

        arg_id = id(arg)
        return arg_id in jit_ctx.func_arg_ids
