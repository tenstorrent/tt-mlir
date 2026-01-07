# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Supported TTNN operations organized by category.

Used by:
- GraphToIRTranslator (graph trace compiler)
- JIT tracing compiler (jit_functions.py)
"""

# Mapping from user-facing op names to TTIR dialect function names.
# Most ops have the same name, but some differ (e.g., divide -> div).
TTIR_NAME_MAP = {
    "divide": "div",
    "pow_tensor": "pow",
}


def get_ttir_name(op_name: str) -> str:
    """Get the TTIR dialect function name for a given operation."""
    return TTIR_NAME_MAP.get(op_name, op_name)


# Unary operations - single input tensor
unary_ops = [
    "abs",
    "exp",
    "neg",
    "sqrt",
    "rsqrt",
    "log",
    "cos",
    "erf",
    "erfc",
    "sign",
    "sin",
    "ceil",
    "floor",
    "tanh",
    "tan",
    "sigmoid",
    "hardsigmoid",
    "relu",
    "reciprocal",
    "gelu",
    "silu",
    "logical_not",
    "bitwise_not",
    "reciprocal",
]

# Binary operations - two input tensors or tensor + scalar
binary_ops = [
    "add",
    "multiply",
    "subtract",
    "divide",
    "pow_tensor",
    "eq",
    "ne",
    "gt",
    "ge",
    "lt",
    "le",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "maximum",
    "minimum",
]

# Reduction operations - operations that reduce tensor dimensions
reduction_ops = [
    "sum",
    "mean",
    "max",
    "min",
]

# Composite operations - operations that should be expanded into internals
# These are operations that are not directly supported but can be decomposed
# into simpler operations that are supported
composite_ops = [
    "digamma",
]

# All supported operations (excluding composite ops that need expansion)
all_ops = set(unary_ops + binary_ops + reduction_ops)


def is_supported(op_name: str) -> bool:
    """Check if an operation is directly supported (not requiring expansion)."""
    return op_name in all_ops


def is_composite(op_name: str) -> bool:
    """Check if an operation is a composite that needs expansion."""
    return op_name in composite_ops


def get_op_category(op_name: str) -> str:
    """Get the category of an operation."""
    if op_name in unary_ops:
        return "unary"
    elif op_name in binary_ops:
        return "binary"
    elif op_name in reduction_ops:
        return "reduction"
    elif op_name in composite_ops:
        return "composite"
    else:
        return "unsupported"
