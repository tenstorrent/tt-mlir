# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Shared operation definitions for TTNN JIT tests.

This module contains all operation wrapper functions used across
test files to avoid repetition and ensure consistency.
"""
import ttnn


# ------------------------------------------------------------
# Unary ops
# ------------------------------------------------------------
def abs(input_tensor):
    return ttnn.abs(input_tensor)


def exp(input_tensor):
    return ttnn.exp(input_tensor)


def log(input_tensor):
    return ttnn.log(input_tensor)


def cos(input_tensor):
    return ttnn.cos(input_tensor)


def sin(input_tensor):
    return ttnn.sin(input_tensor)


def tan(input_tensor):
    return ttnn.tan(input_tensor)


def tanh(input_tensor):
    return ttnn.tanh(input_tensor)


def cbrt(input_tensor):
    return ttnn.cbrt(input_tensor)


def ceil(input_tensor):
    return ttnn.ceil(input_tensor)


def sign(input_tensor):
    return ttnn.sign(input_tensor)


def erf(input_tensor):
    return ttnn.erf(input_tensor)


def erfc(input_tensor):
    return ttnn.erfc(input_tensor)


def floor(input_tensor):
    return ttnn.floor(input_tensor)


def gelu(input_tensor):
    return ttnn.gelu(input_tensor)


def relu(input_tensor):
    return ttnn.relu(input_tensor)


def silu(input_tensor):
    return ttnn.silu(input_tensor)


def logical_not(input_tensor):
    return ttnn.logical_not(input_tensor)


def bitwise_not(input_tensor):
    return ttnn.bitwise_not(input_tensor)


def reciprocal(input_tensor):
    return ttnn.reciprocal(input_tensor)


def sqrt(input_tensor):
    return ttnn.sqrt(input_tensor)


def rsqrt(input_tensor):
    return ttnn.rsqrt(input_tensor)


def sigmoid(input_tensor):
    return ttnn.sigmoid(input_tensor)


def hardsigmoid(input_tensor):
    return ttnn.hardsigmoid(input_tensor)


def neg(input_tensor):
    return ttnn.neg(input_tensor)


# ------------------------------------------------------------
# Binary ops
# ------------------------------------------------------------
def add(a, b):
    return ttnn.add(a, b)


def sub(a, b):
    return ttnn.subtract(a, b)


def mul(a, b):
    return ttnn.multiply(a, b)


def div(a, b):
    return ttnn.divide(a, b)


def logical_and(a, b):
    return ttnn.logical_and(a, b)


def logical_or(a, b):
    return ttnn.logical_or(a, b)


def logical_xor(a, b):
    return ttnn.logical_xor(a, b)


def bitwise_or(a, b):
    return ttnn.bitwise_or(a, b)


def bitwise_and(a, b):
    return ttnn.bitwise_and(a, b)


def bitwise_xor(a, b):
    return ttnn.bitwise_xor(a, b)


# Test pow operation.
#
# Background:
# -----------
# The pow operation had a naming mismatch issue:
# - The Python ttnn API has: ttnn.pow(a, b)
# - The MLIR dialect has: ttnn.pow_tensor (not ttnn.pow)
#
# Original Issue:
# --------------
# PR #5154 changed the test from ttnn.pow() to ttnn.pow_tensor(), but this failed
# because ttnn.pow_tensor doesn't exist in the Python API. When computing the
# golden result (which calls the function directly without JIT), it would error:
#     AttributeError: module 'ttnn' has no attribute 'pow_tensor'
#
# The Fix:
# --------
# 1. Use ttnn.pow() in the test (which exists in Python API)
# 2. Added mapping in graph compiler: "pow" -> "pow_tensor" MLIR op
# 3. Added mapping in AST compiler: node.attr "pow" -> "pow_tensor" MLIR op
#
# Both compilers automatically map ttnn.pow -> ttnn.pow_tensor MLIR operation.
#
# Note: float32 tests may xfail due to numerical precision issues.
def pow(a, b):
    return ttnn.pow(a, b)


def eq(a, b):
    return ttnn.eq(a, b)


def ne(a, b):
    return ttnn.ne(a, b)


def gt(a, b):
    return ttnn.gt(a, b)


def ge(a, b):
    return ttnn.ge(a, b)


def lt(a, b):
    return ttnn.lt(a, b)


def le(a, b):
    return ttnn.le(a, b)


def maximum(a, b):
    return ttnn.maximum(a, b)


def minimum(a, b):
    return ttnn.minimum(a, b)


# ------------------------------------------------------------
# Composite ops
# ------------------------------------------------------------
# Hyperbolic cosine: 0.5 * (exp(x) + exp(-x))
def cosh(input_tensor):
    e_pos_x = ttnn.exp(input_tensor)
    e_neg_x = ttnn.exp(ttnn.neg(input_tensor))
    nr_term = ttnn.add(e_pos_x, e_neg_x)
    output = ttnn.multiply(nr_term, 0.5)
    return output


# Hyperbolic sine: 0.5 * (exp(x) - exp(-x))
def sinh(input_tensor):
    e_pos_x = ttnn.exp(input_tensor)
    e_neg_x = ttnn.exp(ttnn.neg(input_tensor))
    nr_term = ttnn.subtract(e_pos_x, e_neg_x)
    output = ttnn.multiply(nr_term, 0.5)
    return output


# Fused multiply-add: (b * c) + a
def mul_add(input_tensor_a, input_tensor_b, input_tensor_c):
    matmul_result = ttnn.multiply(input_tensor_b, input_tensor_c)
    output = ttnn.add(matmul_result, input_tensor_a)
    return output


# ------------------------------------------------------------
# Other ops
# ------------------------------------------------------------
def matmul(input0, input1):
    return ttnn.matmul(input0, input1)


# Function that uses ttnn.identity, which should be rejected by return_modifier.
def identity_op(input_tensor):
    return ttnn.identity(input_tensor)
