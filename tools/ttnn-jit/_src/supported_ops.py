# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Supported TTNN operations organized by category (Used by GraphToIRTranslator).
"""

# Unary operations - single input tensor
unary_ops = [
    "abs",
    "exp",
    "neg",
    "sqrt",
    "rsqrt",
    "log",
    "cos",
    "sin",
    "ceil",
    "floor",
    "tanh",
    "sigmoid",
    "relu",
    "gelu",
    "silu",
    "logical_not",
    "bitwise_not",
]

# Binary operations - two input tensors or tensor + scalar
binary_ops = [
    "add",
    "multiply",
    "subtract",
    "divide",
    "pow_tensor",
]

# All supported operations
all_ops = set(unary_ops + binary_ops)


def is_supported(op_name: str) -> bool:
    """Check if an operation is supported."""
    return op_name in all_ops


def get_op_category(op_name: str) -> str:
    """Get the category of an operation."""
    if op_name in unary_ops:
        return "unary"
    elif op_name in binary_ops:
        return "binary"
    else:
        return "unsupported"
