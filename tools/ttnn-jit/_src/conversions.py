# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Type and Enum Conversion Utilities.

This module provides unified utilities for converting between different type
and enum representations (TTNN, MLIR, TTCore). All conversions should go through
these functions to maintain consistency across the codebase.

This includes:
- Data type conversions (TTNN <-> MLIR <-> TTCore)
- Buffer type conversions (string -> TTNN enum)
- Memory layout conversions (string -> TTNN enum)
"""
from ttmlir.ir import BF16Type, F32Type, IntegerType
from ttmlir.dialects import ttcore
import ttnn


def mlir_dtype_from_ttnn_dtype(dtype, ctx):
    """
    Convert TTNN dtype to MLIR dtype.

    Args:
        dtype: TTNN dtype (integer enum)
        ctx: MLIR context

    Returns:
        MLIR dtype object
    """

    match dtype.value:
        case 0:
            return BF16Type.get(ctx)
        case 1:
            return F32Type.get(ctx)
        case 2:
            return IntegerType.get_unsigned(32, ctx)
        case 3:
            return ttcore.ir.TileType.get(ctx, 32, 32, ttcore.DataType.BFP_BFloat8)
        case 5:
            return IntegerType.get_unsigned(8, ctx)
        case 6:
            return IntegerType.get_unsigned(16, ctx)
        case 7:
            return IntegerType.get_signless(32, ctx)
        case _:
            raise ValueError(f"Unsupported TTNN dtype: {dtype}")


def ttcore_dtype_from_ttnn_dtype(dtype):
    """
    Convert TTNN dtype to TTCore dtype attribute.

    Args:
        dtype: TTNN dtype (string representation)

    Returns:
        TTCore DataType enum
    """

    match str(dtype):
        case "DataType.BFLOAT16":
            return ttcore.DataType.BFloat16
        case "DataType.FLOAT32":
            return ttcore.DataType.Float32
        case "DataType.INT32":
            return ttcore.DataType.Int32
        case "DataType.BFP_BFloat8":
            return ttcore.DataType.BFP_BFloat8
        case "DataType.BFLOAT8_B":
            return ttcore.DataType.BFP_BFloat8
        case _:
            raise ValueError(f"Unsupported TTNN dtype string: {dtype}")


def ttcore_dtype_from_mlir_dtype(dtype):
    """
    Convert MLIR dtype to TTCore dtype attribute.

    Args:
        dtype: MLIR dtype object

    Returns:
        TTCore DataType enum
    """
    dtype_str = str(dtype)
    match dtype_str:
        case "f32":
            return ttcore.DataType.Float32
        case "bf16":
            return ttcore.DataType.BFloat16
        case s if "bfp_bf8" in s.lower():
            return ttcore.DataType.BFP_BFloat8
        case "i32":
            return ttcore.DataType.Int32
        case _:
            raise ValueError(f"Unsupported MLIR dtype: {dtype}")


def ttnn_dtype_from_mlir_dtype(dtype):
    """
    Convert MLIR dtype to TTNN dtype.

    Args:
        dtype: MLIR dtype object

    Returns:
        TTNN dtype (integer enum)
    """
    dtype_str = str(dtype)
    match dtype_str:
        case "f32":
            return ttnn.DataType.FLOAT32
        case "bf16":
            return ttnn.DataType.BFLOAT16
        case s if "bfp_bf8" in s.lower():
            return ttnn.DataType.BFLOAT8_B
        case "i32":
            return ttnn.DataType.INT32
        case _:
            raise ValueError(f"Unsupported MLIR dtype for TTNN: {dtype}")


def buffer_type_from_string(buffer_type_str: str):
    """
    Convert buffer type string to TTNN BufferType enum.

    Args:
        buffer_type_str: Buffer type as string ("L1" or "DRAM")

    Returns:
        TTNN BufferType enum
    """

    if buffer_type_str == "L1":
        return ttnn.BufferType.L1
    else:
        return ttnn.BufferType.DRAM


def memory_layout_from_string(memory_layout_str: str):
    """
    Convert memory layout string to TTNN TensorMemoryLayout enum.

    Args:
        memory_layout_str: Memory layout as string
            ("BLOCK_SHARDED", "HEIGHT_SHARDED", "WIDTH_SHARDED", or "INTERLEAVED")

    Returns:
        TTNN TensorMemoryLayout enum
    """

    if memory_layout_str == "BLOCK_SHARDED":
        return ttnn.TensorMemoryLayout.BlockSharded
    elif memory_layout_str == "HEIGHT_SHARDED":
        return ttnn.TensorMemoryLayout.HeightSharded
    elif memory_layout_str == "WIDTH_SHARDED":
        return ttnn.TensorMemoryLayout.WidthSharded
    else:
        return ttnn.TensorMemoryLayout.Interleaved


def mlir_memory_layout_from_ttnn_memory_layout(memory_layout):
    """
    Convert TTNN memory layout string representation to TTNN TensorMemoryLayout enum.

    Args:
        memory_layout: TTNN memory layout as string (e.g., "TensorMemoryLayout.INTERLEAVED")

    Returns:
        TTNN TensorMemoryLayout enum
    """
    match str(memory_layout):
        case "TensorMemoryLayout.INTERLEAVED":
            return ttnn.TensorMemoryLayout.Interleaved
        case "TensorMemoryLayout.HEIGHT_SHARDED":
            return ttnn.TensorMemoryLayout.HeightSharded
        case "TensorMemoryLayout.WIDTH_SHARDED":
            return ttnn.TensorMemoryLayout.WidthSharded
        case "TensorMemoryLayout.BLOCK_SHARDED":
            return ttnn.TensorMemoryLayout.BlockSharded
        case _:
            raise ValueError(f"Unsupported memory layout: {memory_layout}")
