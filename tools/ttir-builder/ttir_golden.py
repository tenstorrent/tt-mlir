# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
@file ttir_golden.py
@brief Golden function mappings for TTIR operations.

This module provides a centralized mapping between TTIR operations and their
corresponding PyTorch golden reference implementations.
"""

from typing import Dict, Callable, Any, Optional, Union, List, Tuple
import torch
import torch.nn.functional
from ttmlir.dialects import ttir
from ttmlir.ir import Attribute


def cbrt_golden(x):
    """
    @brief Custom golden function for cubic root.
    @param x Input tensor
    @return Tensor containing the cubic root of each element in the input tensor
    """
    golden_sign = torch.sign(x)
    golden_cbrt = torch.pow(torch.abs(x), 1 / 3)
    return torch.mul(golden_sign, golden_cbrt)


def conv2d_golden(
    input_tensor, weight, bias=None, stride=1, padding=0, dilation=1, groups=1
):
    """
    @brief Custom golden function for conv2d with layout transformation.
    @param input_tensor Input tensor for convolution
    @param weight Convolution weight tensor
    @param bias Optional bias tensor (default: None)
    @param stride Stride for convolution (default: 1)
    @param padding Padding for convolution (default: 0)
    @param dilation Dilation for convolution (default: 1)
    @param groups Number of groups for grouped convolution (default: 1)
    @return Result of 2D convolution with layout transformation
    """
    # ttir can handle a broadcastable bias in the shape [1, 1, 1, C_out], but PyTorch requires the bias is rank 1: [C_out]
    if bias is not None:
        bias = bias.squeeze()  # Removes all dims of size 1

    # Reorganize input and output tensors, golden and ttir functions have different expected tensor shapes
    input_tensor = input_tensor.transpose(-2, -1).transpose(-3, -2)
    result = torch.nn.functional.conv2d(
        input_tensor,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    result = result.transpose(-3, -2).transpose(-2, -1)
    return result


def max_pool2d_golden(input_tensor, kernel_size, stride, padding, dilation, ceil_mode):
    """
    @brief Custom golden function for max_pool2d with layout transformation.
    @param input_tensor Input tensor for max pooling
    @param kernel_size Size of the pooling kernel
    @param stride Stride for pooling operation
    @param padding Padding for pooling operation
    @param dilation Dilation for pooling operation
    @param ceil_mode Whether to use ceiling mode for pooling
    @return Result of 2D max pooling with layout transformation
    """
    # TTIR max_pool2d is channels last. PyTorch max_pool2d is channels first.
    maxpool_object = torch.nn.MaxPool2d(
        kernel_size, stride, padding, dilation, ceil_mode
    )
    input_tensor = input_tensor.transpose(-2, -1).transpose(-3, -2)
    result = maxpool_object(input_tensor)
    result = result.transpose(-3, -2).transpose(-2, -1)
    return result


def argmax_golden(input_tensor, dim_arg, keep_dim=False):
    """
    @brief Custom golden function for argmax.
    @param input_tensor Input tensor to find argmax of
    @param dim_arg List containing dimension to find argmax along
    @param keep_dim Whether to keep the reduced dimension (default: False)
    @return Indices of maximum values along specified dimension as int32 tensor
    """
    result = torch.argmax(input_tensor, dim=dim_arg[0], keepdim=keep_dim)
    return result.to(torch.int32)


def linear_golden(a, b, bias=None, transpose_a=False, transpose_b=False):
    """
    @brief Custom golden function for linear transformation.
    @param a First input tensor
    @param b Second input tensor
    @param bias Optional bias tensor (default: None)
    @param transpose_a Whether to transpose tensor a (default: False)
    @param transpose_b Whether to transpose tensor b (default: False)
    @return Result of linear transformation with optional bias
    """
    a = torch.transpose(a, 0, 1) if transpose_a else a
    b = torch.transpose(b, 0, 1) if transpose_b else b
    output = torch.matmul(a, b)

    if bias is None:
        bias = torch.zeros(list(output.shape))

    bias = (
        torch.broadcast_to(bias, list(output.shape))
        if bias.shape != output.shape
        else bias
    )
    return torch.add(output, bias)


def dot_general_golden(
    lhs, rhs, out, batch_dims_lhs, contract_dims_lhs, batch_dims_rhs, contract_dims_rhs
):
    """
    @brief Custom golden function for dot_general operation.
    @param lhs Left-hand side tensor
    @param rhs Right-hand side tensor
    @param out Output tensor shape reference
    @param batch_dims_lhs Batch dimensions for left tensor
    @param contract_dims_lhs Contraction dimensions for left tensor
    @param batch_dims_rhs Batch dimensions for right tensor
    @param contract_dims_rhs Contraction dimensions for right tensor
    @return Result of generalized dot product operation
    """
    non_batch_dims_lhs = [d for d in range(lhs.dim()) if d not in batch_dims_lhs]
    non_batch_dims_rhs = [d for d in range(rhs.dim()) if d not in batch_dims_rhs]
    transposed_lhs = torch.permute(lhs, (batch_dims_lhs + non_batch_dims_lhs))
    transposed_rhs = torch.permute(rhs, (batch_dims_rhs + non_batch_dims_rhs))
    result = torch.empty(*out.shape, dtype=lhs.dtype)

    dim_ranges = []
    for i in range(len(batch_dims_lhs)):
        dim_ranges.append([j for j in range(list(lhs.shape)[i])])

    import itertools

    batch_indices = list(itertools.product(*dim_ranges))
    for index in batch_indices:
        transposed_lhs_slice = transposed_lhs[index]
        transposed_rhs_slice = transposed_rhs[index]
        dot_dims_lhs = [d - len(index) for d in contract_dims_lhs]
        dot_dims_rhs = [d - len(index) for d in contract_dims_rhs]
        out_index = index
        result[out_index] = torch.tensordot(
            transposed_lhs_slice,
            transposed_rhs_slice,
            dims=(dot_dims_lhs, dot_dims_rhs),
        )
    return result


def quantize_golden(input_tensor, scale, zero_point, dtype):
    """
    @brief Custom golden function for quantize operation.
    @param input_tensor Input tensor to quantize
    @param scale Scale factor for quantization
    @param zero_point Zero point for quantization
    @param dtype Target quantized data type
    @return Quantized tensor as integer representation
    """
    return torch.quantize_per_tensor(input_tensor, scale, zero_point, dtype).int_repr()


def requantize_golden(input_tensor, scale, zero_point, dtype):
    """
    @brief Custom golden function for requantize operation.
    @param input_tensor Input quantized tensor to requantize
    @param scale Scale factor for requantization
    @param zero_point Zero point for requantization
    @param dtype Target quantized data type
    @return Requantized tensor
    """
    return torch.quantize_per_tensor(
        torch.dequantize(input_tensor), scale, zero_point, dtype
    )


def max_golden(input_tensor, dim_arg=None, keep_dim=True):
    """
    @brief Custom golden function for max operation with conditional logic.
    @param input_tensor Input tensor to find maximum of
<<<<<<< HEAD
    @param dim_arg Dimension to find maximum along (can be None, int, or list with single int)
=======
    @param dim_arg Dimension to find maximum along (default: None for all dimensions)
>>>>>>> 1e49ed63c (all golden functions extracted into ttir_golden and ops.py refactored)
    @param keep_dim Whether to keep the reduced dimension (default: True)
    @return Maximum values along specified dimension or global maximum
    """
    if dim_arg is not None:
<<<<<<< HEAD
        # Handle case where dim_arg is passed as a list [dim_arg]
        if isinstance(dim_arg, list) and len(dim_arg) == 1:
            dim_arg = dim_arg[0]
=======
>>>>>>> 1e49ed63c (all golden functions extracted into ttir_golden and ops.py refactored)
        return torch.max(input_tensor, dim=dim_arg, keepdim=keep_dim)
    else:
        # For all dimensions reduction, reshape to match expected output
        result = torch.max(input_tensor)
        output_shape = [1] * input_tensor.dim()
        return result.reshape(*output_shape)


def prod_golden(input_tensor, dim_arg, keep_dim=False):
    """
    @brief Custom golden function for prod operation with conditional logic.
    @param input_tensor Input tensor to compute product of
    @param dim_arg List of dimensions to compute product along
    @param keep_dim Whether to keep the reduced dimension (default: False)
    @return Product of tensor elements along specified dimensions
    """
    if len(dim_arg) == 1:
        return torch.prod(input_tensor, dim=dim_arg[0], keepdim=keep_dim)
    else:
        # Multiple dimensions - reduce to scalar
        return torch.tensor([torch.prod(input_tensor).item()])


def embedding_golden(indices_tensor, weight_tensor):
    """
    @brief Custom golden function for embedding operation.
    @param indices_tensor Tensor containing indices to look up
    @param weight_tensor Weight tensor containing embedding vectors
    @return Embedded vectors corresponding to input indices
    """
    embedding = torch.nn.Embedding.from_pretrained(weight_tensor)
    golden_typecast = indices_tensor.to(torch.int32)
    golden_input = torch.clamp(golden_typecast, 0, (weight_tensor.size()[0] - 1))
    return embedding(golden_input)


def pad_golden(input_tensor, padding, value):
    """
    @brief Custom golden function for pad operation with dimension reformatting.
    @param input_tensor Input tensor to pad
    @param padding Padding specification
    @param value Value to use for padding
    @return Padded tensor
    """
    # Reformatting padding dimensions for golden tensor:
    golden_padding = []
    for i in range(len(padding) // 2):
        golden_padding.append(padding[-((2 * i) + 2)])
        golden_padding.append(padding[-((2 * i) + 1)])
    return torch.nn.functional.pad(
        input_tensor, pad=golden_padding, mode="constant", value=value
    )


def select_golden(input_tensor, dim, begin, length, stride):
    """
    @brief Custom golden function for select operation.
    @param input_tensor Input tensor to select from
    @param dim Dimension to select along
    @param begin Starting index for selection
    @param length Length of selection
    @param stride Stride for selection
    @return Selected tensor slice
    """
    end = begin + length - 1
    index = torch.tensor([begin, end])
    return torch.index_select(input_tensor, dim=dim, index=index)


def index_golden(input_tensor, dim, begin, end, step):
    """
    @brief Custom golden function for index operation.
    @param input_tensor Input tensor to index
    @param dim Dimension to index along
    @param begin Starting index
    @param end Ending index
    @param step Step size for indexing
    @return Indexed tensor
    """
    import math

    num_indices = math.ceil((end - begin) / step)
    indices = []
    for i in range(num_indices):
        indices.append((begin + i) * step)
    index = torch.tensor(indices)
    return torch.index_select(input_tensor, dim=dim, index=index)


def arange_golden(single_dim_tensor, repeats):
    """
    @brief Custom golden function for arange operation.
    @param single_dim_tensor Single dimension tensor specification
    @param repeats Number of repeats for the range
    @return Generated range tensor
    """
    return single_dim_tensor.repeat(repeats)


def slice_golden(input_tensor, begins, ends, step):
    """
    @brief Custom golden function for slice operation.
    @param input_tensor Input tensor to slice
    @param begins Starting indices for each dimension
    @param ends Ending indices for each dimension
    @param step Step sizes for each dimension
    @return Sliced tensor
    """
    # Build slice objects for each dimension
    slices = []
    for i, (b, e, s) in enumerate(zip(begins, ends, step)):
        slices.append(slice(b, e, s))

    # Apply slicing to the tensor
    return input_tensor[tuple(slices)]


def gather_golden(
    input_tensor,
    start_indices_tensor,
    offset_dims,
    collapsed_slice_dims,
    operand_batching_dims,
    start_indices_batching_dims,
    start_index_map,
    index_vector_dim,
    slice_sizes,
    indices_are_sorted=False,
):
    """
    @brief Custom golden function for gather operation.
    @param input_tensor Input tensor to gather from
    @param start_indices_tensor Tensor containing starting indices
    @param offset_dims Offset dimensions for gathering
    @param collapsed_slice_dims Dimensions to collapse after slicing
    @param operand_batching_dims Batching dimensions for operand
    @param start_indices_batching_dims Batching dimensions for start indices
    @param start_index_map Mapping of start indices
    @param index_vector_dim Dimension containing index vectors
    @param slice_sizes Sizes of slices to gather
    @param indices_are_sorted Whether indices are sorted (default: False)
    @return Gathered tensor
    """
    # Simple gather implementation for basic cases
    if (
        len(offset_dims) == 1
        and offset_dims[0] == 1
        and len(collapsed_slice_dims) == 1
        and collapsed_slice_dims[0] == 0
        and len(operand_batching_dims) == 0
        and len(start_indices_batching_dims) == 0
        and len(start_index_map) == 1
        and start_index_map[0] == 0
        and index_vector_dim == 1
        and len(slice_sizes) == 2
        and slice_sizes[0] == 1
    ):

        indices = start_indices_tensor.squeeze().long()
        device = input_tensor.device if hasattr(input_tensor, "device") else None

        if indices.dim() == 0:
            indices = indices.unsqueeze(0)

        output = []
        for idx in indices:
            output.append(input_tensor[idx, : slice_sizes[1]])

        output = torch.stack(output)
        return torch.tensor(output, device=device)
    else:
        # General gather case (not implemented)
        raise NotImplementedError("General gather not implemented")


def tilize_golden(input_tensor):
    """
    @brief Custom golden function for tilize operation.
    @param input_tensor Input tensor to tilize
    @return Tilized tensor with proper tile layout transformation
    """
    shape = input_tensor.shape
    TILE_SIZE = 32
    FACE_SIZE = 16
    Y_TILES = shape[0] // TILE_SIZE
    X_TILES = shape[1] // TILE_SIZE
    FACES_PER_TILE = TILE_SIZE // FACE_SIZE

    tilized = torch.zeros((input_tensor.numel(),))

    idx = 0
    for tile_y in range(Y_TILES):
        for tile_x in range(X_TILES):
            for face_y in range(FACES_PER_TILE):
                for face_x in range(FACES_PER_TILE):
                    for datum_y in range(FACE_SIZE):
                        for datum_x in range(FACE_SIZE):
                            tilized[idx] = input_tensor[
                                datum_y + tile_y * TILE_SIZE + face_y * FACE_SIZE,
                                datum_x + tile_x * TILE_SIZE + face_x * FACE_SIZE,
                            ]
                            idx += 1

    tilized = tilized.reshape(shape)
    return tilized


def untilize_golden(input_tensor):
    """
    @brief Custom golden function for untilize operation.
    @param input_tensor Input tensor to untilize
    @return Untilized tensor with proper layout transformation
    """
    shape = input_tensor.shape
    TILE_SIZE = 32
    FACE_SIZE = 16
    Y_TILES = shape[0] // TILE_SIZE
    X_TILES = shape[1] // TILE_SIZE
    FACES_PER_TILE = TILE_SIZE // FACE_SIZE

    untilized = torch.zeros_like(input_tensor)
    flattened = input_tensor.flatten()

    idx = 0
    for tile_y in range(Y_TILES):
        for tile_x in range(X_TILES):
            for face_y in range(FACES_PER_TILE):
                for face_x in range(FACES_PER_TILE):
                    for datum_y in range(FACE_SIZE):
                        for datum_x in range(FACE_SIZE):
                            # Calculate the original position
                            orig_y = datum_y + tile_y * TILE_SIZE + face_y * FACE_SIZE
                            orig_x = datum_x + tile_x * TILE_SIZE + face_x * FACE_SIZE

                            # Place the value from the tilized tensor back to its original position
                            untilized[orig_y, orig_x] = flattened[idx]
                            idx += 1

    return untilized


def upsample2d_golden(in0, in1, scale_factor, mode="nearest"):
    """
    @brief Custom golden function for upsample2d operation.
    @param in0 Input tensor to upsample
    @param in1 Output tensor specification
    @param scale_factor Scaling factor for upsampling
    @param mode Upsampling mode (default: "nearest")
    @return Upsampled 2D tensor
    """
    transposed_golden = torch.transpose(in0, 1, 3)
    golden_output_shape = in1.shape[1:-1]
    output = torch.nn.functional.interpolate(
        transposed_golden, size=golden_output_shape, mode=mode
    )
    return torch.transpose(output, 1, 3)


def fill_cache_golden(cache_tensor, input_tensor):
    """
    @brief Custom golden function for fill_cache operation.
    @param cache_tensor Cache tensor to fill
    @param input_tensor Input tensor data
    @return Filled cache tensor
    """
    result = cache_tensor.clone()
    result[:, :, : input_tensor.shape[2], :] = input_tensor
    return result


def update_cache_golden(cache_tensor, update_tensor, indices_tensor):
    """
    @brief Custom golden function for update_cache operation.
    @param cache_tensor Cache tensor to update
    @param update_tensor Tensor containing update data
    @param indices_tensor Tensor containing update indices
    @return Updated cache tensor
    """
    result = cache_tensor.clone()
    # Simple update logic - this would need to be refined based on actual requirements
    result[:, :, : update_tensor.shape[2], :] = update_tensor
    return result


def mesh_shard_golden(
    input: torch.Tensor,
    mesh_shape: Tuple[int, int],
    shard_type: Attribute,
    shard_direction: Attribute,
    shard_shape: Tuple[int, int],
    shard_dims: List[int],
) -> torch.Tensor:
    """
    @brief Return a random torch.Tensor which has the correct shape and type after doing mesh_shard on the input.
    @param input Input tensor to be sharded
    @param mesh_shape Shape of the device mesh
    @param shard_type Type of sharding operation
    @param shard_direction Direction of sharding
    @param shard_shape Shape of the shard
    @param shard_dims Dimensions to shard along
    @return Random tensor with correct output shape and type
    """
    out_shape = list(input.shape)
    if "devices" in str(shard_type).lower():
        for shard_dim in shard_dims:
            if shard_dim == -1:
                continue
            if "shard_to_full" in str(shard_direction).lower():
                out_shape[shard_dim] *= shard_shape[shard_dim]
            elif "full_to_shard" in str(shard_direction).lower():
                out_shape[shard_dim] //= shard_shape[shard_dim]
    return torch.randn(out_shape, dtype=input.dtype)


def all_gather_golden(
    input: torch.Tensor,
    mesh_shape: Tuple[int, int],
    all_gather_dim: int,
    cluster_axis: int,
) -> torch.Tensor:
    """
    @brief Return a random torch.Tensor which has the correct shape and type after doing all_gather on the input.
    @param input Input tensor to gather from all devices
    @param mesh_shape Shape of the device mesh
    @param all_gather_dim Dimension to gather along
    @param cluster_axis Axis of the cluster for gathering
    @return Random tensor with correct output shape and type
    """
    out_shape = list(input.shape)
    out_shape[all_gather_dim] *= mesh_shape[cluster_axis]
    return torch.randn(out_shape, dtype=input.dtype)


def all_reduce_golden(
    input: torch.Tensor,
    mesh_shape: Tuple[int, int],
    cluster_axis: int,
    reduce_type: Attribute,
) -> torch.Tensor:
    """
    @brief Return a random torch.Tensor which has the correct shape and type after doing all_reduce on the input.
    @param input Input tensor to reduce across devices
    @param mesh_shape Shape of the device mesh
    @param cluster_axis Axis of the cluster for reduction
    @param reduce_type Type of reduction operation
    @return Random tensor with correct output shape and type
    """
    return torch.randn(input.shape, dtype=input.dtype)


def reduce_scatter_golden(
    input: torch.Tensor,
    mesh_shape: Tuple[int, int],
    reduce_type: Attribute,
    scatter_dim: int,
    cluster_axis: int,
) -> torch.Tensor:
    """
    @brief Return a random torch.Tensor which has the correct shape and type after doing reduce_scatter on the input.
    @param input Input tensor to reduce and scatter
    @param mesh_shape Shape of the device mesh
    @param reduce_type Type of reduction operation
    @param scatter_dim Dimension to scatter along
    @param cluster_axis Axis of the cluster for operation
    @return Random tensor with correct output shape and type
    """
    out_shape = list(input.shape)
    out_shape[scatter_dim] //= mesh_shape[cluster_axis]
    return torch.randn(out_shape, dtype=input.dtype)


def collective_permute_golden(
    input: torch.Tensor,
    mesh_shape: Tuple[int, int],
    source_target_pairs: List[Tuple[int, int]],
) -> torch.Tensor:
    """
    @brief Return a random torch.Tensor which has the correct shape and type after doing collective_permute on the input.
    @param input Input tensor to permute across devices
    @param mesh_shape Shape of the device mesh
    @param source_target_pairs List of (source, target) device ID pairs for permutation
    @return Random tensor with correct output shape and type
    """
    return torch.randn(input.shape, dtype=input.dtype)


<<<<<<< HEAD
def create_smart_golden_wrapper(original_func, convert_kwargs=None):
    """
    @brief Create a wrapper for golden functions that can accept TTIR-style arguments.
    @param original_func The original golden function that expects tensors
    @param convert_kwargs Dictionary mapping TTIR kwarg names to golden function kwarg names
    @return Wrapper function that accepts TTIR arguments and converts to PyTorch format
    """

    def wrapper(*args, **kwargs):
        def convert_mlir_value(value):
            """Convert MLIR attributes to Python values"""
            import torch

            # Don't convert PyTorch tensors - they should stay as tensors
            if isinstance(value, torch.Tensor):
                return value

            # Handle DenseI32ArrayAttr, DenseI64ArrayAttr, etc.
            if hasattr(value, "__iter__") and hasattr(value, "_CAPIPtr"):
                try:
                    # Convert to list and recursively convert elements
                    result_list = list(value)
                    converted_list = [convert_mlir_value(item) for item in result_list]
                    # Ensure we return plain Python integers, not numpy/torch types
                    return [
                        int(item) if hasattr(item, "__index__") else item
                        for item in converted_list
                    ]
                except:
                    return value
            # Handle IntegerAttr
            elif hasattr(value, "value") and hasattr(value, "_CAPIPtr"):
                try:
                    result = value.value
                    # Ensure we return plain Python int, not numpy/torch types
                    return int(result) if hasattr(result, "__index__") else result
                except:
                    return value
            # Handle regular lists that might contain MLIR attributes
            elif isinstance(value, (list, tuple)):
                converted_list = [convert_mlir_value(item) for item in value]
                # Ensure we return plain Python integers, not numpy/torch types
                return [
                    int(item) if hasattr(item, "__index__") else item
                    for item in converted_list
                ]
            else:
                return value

        if convert_kwargs:
            # Convert TTIR-style kwargs to golden function format
            converted_kwargs = {}
            for key, value in kwargs.items():
                converted_value = convert_mlir_value(value)

                if key in convert_kwargs:
                    # Use the mapping to convert kwarg names
                    golden_key = convert_kwargs[key]
                    if (
                        golden_key is not None
                    ):  # Only add if not None (for filtering out params)
                        converted_kwargs[golden_key] = converted_value
                else:
                    # Keep unmapped kwargs as-is (but converted)
                    converted_kwargs[key] = converted_value
            return original_func(*args, **converted_kwargs)
        else:
            # No conversion needed, but still convert MLIR attributes
            converted_kwargs = {}
            for key, value in kwargs.items():
                converted_kwargs[key] = convert_mlir_value(value)
            return original_func(*args, **converted_kwargs)

    return wrapper


# Smart wrappers for common reduction operations
def sum_ttir_compatible(input_tensor, dim_arg=None, keep_dim=True):
    """
    @brief TTIR-compatible wrapper for torch.sum
    @param input_tensor Input tensor to sum
    @param dim_arg TTIR-style dimension argument (list or None)
    @param keep_dim Whether to keep reduced dimensions
    @return Sum along specified dimensions
    """
    if dim_arg is None:
        return torch.sum(input_tensor, keepdim=keep_dim)
    elif isinstance(dim_arg, list):
        return torch.sum(input_tensor, dim=dim_arg, keepdim=keep_dim)
    else:
        return torch.sum(input_tensor, dim=dim_arg, keepdim=keep_dim)


def mean_ttir_compatible(input_tensor, dim_arg=None, keep_dim=True):
    """
    @brief TTIR-compatible wrapper for torch.mean
    @param input_tensor Input tensor to compute mean of
    @param dim_arg TTIR-style dimension argument (list or None)
    @param keep_dim Whether to keep reduced dimensions
    @return Mean along specified dimensions
    """
    if dim_arg is None:
        return torch.mean(input_tensor, keepdim=keep_dim)
    elif isinstance(dim_arg, list):
        return torch.mean(input_tensor, dim=dim_arg, keepdim=keep_dim)
    else:
        return torch.mean(input_tensor, dim=dim_arg, keepdim=keep_dim)


def min_ttir_compatible(input_tensor, dim_arg=None, keep_dim=True):
    """
    @brief TTIR-compatible wrapper for torch.min
    @param input_tensor Input tensor to find minimum of
    @param dim_arg TTIR-style dimension argument (can be None, int, or list with single int)
    @param keep_dim Whether to keep reduced dimensions
    @return Minimum values along specified dimension
    """
    if dim_arg is None:
        # Global minimum
        result = torch.min(input_tensor)
        if keep_dim:
            # Reshape to maintain all dimensions as size 1
            output_shape = [1] * input_tensor.dim()
            return result.reshape(*output_shape)
        else:
            return result
    else:
        # Handle case where dim_arg is passed as a list [dim_arg]
        if isinstance(dim_arg, list) and len(dim_arg) == 1:
            dim_arg = dim_arg[0]
        return torch.min(input_tensor, dim=dim_arg, keepdim=keep_dim)


def transpose_ttir_compatible(input_tensor, dim0, dim1):
    """
    @brief TTIR-compatible wrapper for torch.transpose
    @param input_tensor Input tensor to transpose
    @param dim0 First dimension to transpose
    @param dim1 Second dimension to transpose
    @return Transposed tensor
    """
    return torch.transpose(input_tensor, dim0, dim1)


def permute_ttir_compatible(input_tensor, permutation):
    """
    @brief TTIR-compatible wrapper for torch.permute that accepts permutation as kwarg
    @param input_tensor Input tensor to permute
    @param permutation List of dimension indices for permutation (can be MLIR attribute or list)
    @return Permuted tensor
    """
    # Ensure permutation is a tuple of ints for torch.permute
    if isinstance(permutation, (list, tuple)):
        permutation = tuple(int(dim) for dim in permutation)
    return torch.permute(input_tensor, permutation)


def reverse_ttir_compatible(input_tensor, dimensions):
    """
    @brief TTIR-compatible wrapper for torch.flip (reverse)
    @param input_tensor Input tensor to reverse
    @param dimensions Dimensions to reverse along (TTIR-style argument name)
    @return Tensor with reversed elements
    """
    return torch.flip(input_tensor, dims=dimensions)


def squeeze_ttir_compatible(input_tensor, dim=None):
    """
    @brief TTIR-compatible wrapper for torch.squeeze
    @param input_tensor Input tensor to squeeze
    @param dim Dimension to squeeze (can be None)
    @return Squeezed tensor
    """
    if dim is None:
        return torch.squeeze(input_tensor)
    else:
        return torch.squeeze(input_tensor, dim=dim)


def unsqueeze_ttir_compatible(input_tensor, dim=0):
    """
    @brief TTIR-compatible wrapper for torch.unsqueeze
    @param input_tensor Input tensor to unsqueeze
    @param dim Dimension to add
    @return Unsqueezed tensor
    """
    return torch.unsqueeze(input_tensor, dim=dim)


def broadcast_ttir_compatible(input_tensor, size):
    """
    @brief TTIR-compatible wrapper for torch.broadcast_to
    @param input_tensor Input tensor to broadcast
    @param size Target size for broadcasting
    @return Broadcasted tensor
    """
    return torch.broadcast_to(input_tensor, size)


def softmax_ttir_compatible(input_tensor, dimension=1):
    """
    @brief TTIR-compatible wrapper for torch.nn.functional.softmax
    @param input_tensor Input tensor for softmax
    @param dimension Dimension to apply softmax along
    @return Softmax result
    """
    return torch.nn.functional.softmax(input_tensor, dim=dimension)


def slice_ttir_compatible(input_tensor, begins, ends, step):
    """
    @brief TTIR-compatible wrapper for slice operation that accepts MLIR attributes or raw values
    @param input_tensor Input tensor to slice
    @param begins Starting indices (can be MLIR attributes or raw values)
    @param ends Ending indices (can be MLIR attributes or raw values)
    @param step Step sizes (can be MLIR attributes or raw values)
    @return Sliced tensor
    """
    # Handle MLIR attributes by extracting raw values
    from ttmlir.ir import ArrayAttr

    if hasattr(begins, "value") and hasattr(begins, "__getitem__"):  # MLIR ArrayAttr
        begins_values = [attr.value for attr in begins]
    else:
        begins_values = begins

    if hasattr(ends, "value") and hasattr(ends, "__getitem__"):  # MLIR ArrayAttr
        ends_values = [attr.value for attr in ends]
    else:
        ends_values = ends

    if hasattr(step, "value") and hasattr(step, "__getitem__"):  # MLIR ArrayAttr
        step_values = [attr.value for attr in step]
    else:
        step_values = step

    # Use the original slice_golden implementation
    return slice_golden(input_tensor, begins_values, ends_values, step_values)


def max_pool2d_ttir_compatible(
    input_tensor,
    kernel_height,
    kernel_width,
    stride_height,
    stride_width,
    dilation_height,
    dilation_width,
    ceil_mode,
    padding_left,
    padding_right,
    padding_top,
    padding_bottom,
):
    """
    @brief TTIR-compatible wrapper for 2D max pooling that accepts individual parameters.
    @param input_tensor Input tensor for max pooling
    @param kernel_height Height of pooling kernel
    @param kernel_width Width of pooling kernel
    @param stride_height Stride height
    @param stride_width Stride width
    @param dilation_height Dilation height
    @param dilation_width Dilation width
    @param ceil_mode Whether to use ceiling mode
    @param padding_left Left padding
    @param padding_right Right padding
    @param padding_top Top padding
    @param padding_bottom Bottom padding
    @return Max pooled tensor with layout conversion
    """
    kernel_size = (kernel_height, kernel_width)
    stride = (stride_height, stride_width)
    padding = (
        padding_top,
        padding_left,
    )  # PyTorch expects (top, left) for asymmetric padding
    dilation = (dilation_height, dilation_width)

    # TTIR max_pool2d is channels last. PyTorch max_pool2d is channels first.
    maxpool_object = torch.nn.MaxPool2d(
        kernel_size, stride, padding, dilation, ceil_mode
    )
    input_tensor = input_tensor.transpose(-2, -1).transpose(-3, -2)
    result = maxpool_object(input_tensor)
    result = result.transpose(-3, -2).transpose(-2, -1)
    return result


def linear_ttir_compatible(a, b, transpose_a=False, transpose_b=False, bias=None):
    """
    @brief TTIR-compatible wrapper for linear operation
    @param a First input tensor
    @param b Second input tensor
    @param transpose_a Whether to transpose first tensor
    @param transpose_b Whether to transpose second tensor
    @param bias Optional bias tensor
    @return Result of linear transformation
    """
    a = torch.transpose(a, 0, 1) if transpose_a else a
    b = torch.transpose(b, 0, 1) if transpose_b else b
    output = torch.matmul(a, b)
    if bias is not None:
        bias = (
            bias if isinstance(bias, torch.Tensor) else torch.zeros(list(output.shape))
        )
        bias = (
            torch.broadcast_to(bias, list(output.shape))
            if bias.shape != output.shape
            else bias
        )
        output = torch.add(output, bias)
    return output


=======
>>>>>>> 1e49ed63c (all golden functions extracted into ttir_golden and ops.py refactored)
## @brief Dictionary mapping TTIR operation classes to their corresponding golden functions.
##
## This dictionary provides a centralized mapping between TTIR operation types and their
## PyTorch-based golden reference implementations. Each key is a TTIR operation class
## (e.g., ttir.AbsOp) and each value is the corresponding golden function that computes
## the expected output for that operation.
GOLDEN_MAPPINGS: Dict[type, Callable] = {
<<<<<<< HEAD
    # Elementwise unary operations - using smart wrappers
    ttir.GetDimensionSizeOp: create_smart_golden_wrapper(
        torch.tensor,
        convert_kwargs={"dimension": None},  # Filter out dimension parameter
    ),
    ttir.AbsOp: create_smart_golden_wrapper(torch.abs),
    ttir.CeilOp: create_smart_golden_wrapper(torch.ceil),
    ttir.CosOp: create_smart_golden_wrapper(torch.cos),
    ttir.FloorOp: create_smart_golden_wrapper(torch.floor),
    ttir.GeluOp: create_smart_golden_wrapper(torch.nn.functional.gelu),
    ttir.IsFiniteOp: create_smart_golden_wrapper(torch.isfinite),
    ttir.NegOp: create_smart_golden_wrapper(torch.neg),
    ttir.TanOp: create_smart_golden_wrapper(torch.tan),
    ttir.AtanOp: create_smart_golden_wrapper(torch.atan),
    ttir.TanhOp: create_smart_golden_wrapper(torch.tanh),
    ttir.ReciprocalOp: create_smart_golden_wrapper(torch.reciprocal),
    ttir.ReluOp: create_smart_golden_wrapper(torch.relu),
    ttir.RsqrtOp: create_smart_golden_wrapper(torch.rsqrt),
    ttir.SigmoidOp: create_smart_golden_wrapper(torch.sigmoid),
    ttir.SignOp: create_smart_golden_wrapper(torch.sign),
    ttir.SinOp: create_smart_golden_wrapper(torch.sin),
    ttir.SqrtOp: create_smart_golden_wrapper(torch.sqrt),
    ttir.LogOp: create_smart_golden_wrapper(torch.log),
    ttir.Log1pOp: create_smart_golden_wrapper(torch.log1p),
    ttir.Expm1Op: create_smart_golden_wrapper(torch.expm1),
    ttir.ExpOp: create_smart_golden_wrapper(torch.exp),
    # Elementwise binary operations - using smart wrappers
    ttir.AddOp: create_smart_golden_wrapper(torch.add),
    ttir.MultiplyOp: create_smart_golden_wrapper(torch.multiply),
    ttir.SubtractOp: create_smart_golden_wrapper(torch.subtract),
    ttir.DivOp: create_smart_golden_wrapper(torch.div),
    ttir.MaximumOp: create_smart_golden_wrapper(torch.maximum),
    ttir.MinimumOp: create_smart_golden_wrapper(torch.minimum),
    ttir.RemainderOp: create_smart_golden_wrapper(torch.remainder),
    ttir.PowOp: create_smart_golden_wrapper(torch.pow),
    # Comparison operations - using smart wrappers
    ttir.EqualOp: create_smart_golden_wrapper(torch.eq),
    ttir.NotEqualOp: create_smart_golden_wrapper(torch.ne),
    ttir.GreaterEqualOp: create_smart_golden_wrapper(torch.ge),
    ttir.GreaterThanOp: create_smart_golden_wrapper(torch.gt),
    ttir.LessEqualOp: create_smart_golden_wrapper(torch.le),
    ttir.LessThanOp: create_smart_golden_wrapper(torch.lt),
    # Logical operations - using smart wrappers
    ttir.LogicalAndOp: create_smart_golden_wrapper(torch.logical_and),
    ttir.LogicalOrOp: create_smart_golden_wrapper(torch.logical_or),
    ttir.LogicalXorOp: create_smart_golden_wrapper(torch.logical_xor),
    ttir.LogicalNotOp: create_smart_golden_wrapper(torch.logical_not),
    # Selection operations - using smart wrappers
    ttir.WhereOp: create_smart_golden_wrapper(torch.where),
    # Bitwise operations - using smart wrappers
    ttir.BitwiseAndOp: create_smart_golden_wrapper(torch.bitwise_and),
    ttir.BitwiseOrOp: create_smart_golden_wrapper(torch.bitwise_or),
    ttir.BitwiseXorOp: create_smart_golden_wrapper(torch.bitwise_xor),
    ttir.BitwiseNotOp: create_smart_golden_wrapper(torch.bitwise_not),
    # Reduction operations - using smart wrappers with parameter conversions
    ttir.SumOp: create_smart_golden_wrapper(
        torch.sum, convert_kwargs={"dim_arg": "dim", "keep_dim": "keepdim"}
    ),
    ttir.MeanOp: create_smart_golden_wrapper(
        torch.mean, convert_kwargs={"dim_arg": "dim", "keep_dim": "keepdim"}
    ),
    ttir.MaxOp: max_golden,  # Keep as-is - has complex parameter conversion logic
    ttir.MinOp: create_smart_golden_wrapper(min_ttir_compatible),
    ttir.ProdOp: create_smart_golden_wrapper(
        lambda input, dim, keepdim=False, **kwargs: torch.prod(
            input,
            dim=dim[0]
            if isinstance(dim, list) and len(dim) == 1
            else tuple(dim)
            if isinstance(dim, list)
            else dim,
            keepdim=keepdim,
        ),
        convert_kwargs={"dim_arg": "dim", "keep_dim": "keepdim"},
    ),
    ttir.ReduceAndOp: create_smart_golden_wrapper(
        torch.all, convert_kwargs={"dim_arg": "dim", "keep_dim": "keepdim"}
    ),
    ttir.ReduceOrOp: create_smart_golden_wrapper(
        torch.any, convert_kwargs={"dim_arg": "dim", "keep_dim": "keepdim"}
    ),
    # Tensor manipulation - using smart wrappers with parameter conversions
    ttir.TransposeOp: create_smart_golden_wrapper(torch.transpose),
    ttir.ConcatOp: create_smart_golden_wrapper(torch.concat),
    ttir.RepeatOp: create_smart_golden_wrapper(
        torch.Tensor.repeat, convert_kwargs={"repeat_dimensions": "repeats"}
    ),
    ttir.RepeatInterleaveOp: create_smart_golden_wrapper(torch.repeat_interleave),
    ttir.ReshapeOp: create_smart_golden_wrapper(torch.reshape),
    ttir.SqueezeOp: squeeze_ttir_compatible,
    ttir.UnsqueezeOp: unsqueeze_ttir_compatible,
    ttir.ReverseOp: reverse_ttir_compatible,
    ttir.PermuteOp: create_smart_golden_wrapper(permute_ttir_compatible),
    ttir.ClampScalarOp: create_smart_golden_wrapper(torch.clamp),
    ttir.ClampTensorOp: torch.clamp,  # Keep as-is (needs special handling for multiple operands)
    ttir.BroadcastOp: create_smart_golden_wrapper(
        lambda input_tensor, size, **kwargs: torch.broadcast_to(input_tensor, size),
        convert_kwargs={
            "broadcast_dimensions": None
        },  # Filter out broadcast_dimensions
    ),
    ttir.PadOp: create_smart_golden_wrapper(pad_golden),
    ttir.IndexSelectOp: select_golden,
    ttir.IndexOp: index_golden,
    ttir.SliceOp: create_smart_golden_wrapper(
        slice_golden,
        convert_kwargs={},  # Let the wrapper handle MLIR attribute conversion
    ),
    ttir.GatherOp: create_smart_golden_wrapper(
        gather_golden,
        convert_kwargs={},  # Let the wrapper handle MLIR attribute conversion
    ),
    # Neural network operations - using TTIR-compatible wrappers
    ttir.SoftmaxOp: create_smart_golden_wrapper(
        softmax_ttir_compatible,
        convert_kwargs={
            "dimension": "dimension"
        },  # Keep dimension as dimension for softmax_ttir_compatible
    ),
    ttir.MatmulOp: create_smart_golden_wrapper(torch.matmul),
    ttir.EmbeddingOp: create_smart_golden_wrapper(embedding_golden),
    ttir.CumSumOp: create_smart_golden_wrapper(
        torch.cumsum,
        convert_kwargs={"output": None},  # Remove output parameter for golden function
    ),
    ttir.Upsample2dOp: upsample2d_golden,
    # Type operations - using smart wrappers
    ttir.TypecastOp: create_smart_golden_wrapper(
        torch.Tensor.type, convert_kwargs={}  # Just passes dtype through
    ),
    # Tensor creation - using smart wrappers where appropriate
    ttir.ZerosOp: create_smart_golden_wrapper(
        torch.zeros,
        convert_kwargs={
            "result": None,
            "shape": "size",
        },  # Filter out result, rename shape to size
    ),
    ttir.OnesOp: create_smart_golden_wrapper(
        torch.ones,
        convert_kwargs={
            "result": None,
            "shape": "size",
        },  # Filter out result, rename shape to size
    ),
    ttir.ArangeOp: create_smart_golden_wrapper(
        arange_golden,
        convert_kwargs={
            "start": None,
            "end": None,
            "step": None,
            "arange_dimension": None,
        },  # Filter out parameters not needed by golden function
    ),
    # Quantization operations - using smart wrappers where appropriate
    ttir.QuantizeOp: create_smart_golden_wrapper(quantize_golden),
    ttir.DequantizeOp: create_smart_golden_wrapper(torch.dequantize),
    ttir.RequantizeOp: create_smart_golden_wrapper(requantize_golden),
    # Complex operations - using TTIR-compatible wrappers
    ttir.CbrtOp: create_smart_golden_wrapper(cbrt_golden),
    ttir.Conv2dOp: create_smart_golden_wrapper(
        conv2d_golden,
        convert_kwargs={},  # Let the wrapper handle MLIR attribute conversion
    ),
    ttir.MaxPool2dOp: create_smart_golden_wrapper(
        max_pool2d_ttir_compatible,
        convert_kwargs={},  # Let the wrapper handle parameter mapping
    ),
    ttir.ArgMaxOp: argmax_golden,
    ttir.LinearOp: linear_ttir_compatible,
    ttir.DotGeneralOp: dot_general_golden,
    # Layout operations (identity functions)
    ttir.ToLayoutOp: lambda x, **kwargs: x,
    ttir.ViewLayoutOp: lambda x, **kwargs: x,
    # Cache operations - using smart wrappers
    ttir.FillCacheOp: create_smart_golden_wrapper(
        fill_cache_golden,
        convert_kwargs={"batch_offset": None},  # Filter out batch_offset parameter
    ),
    ttir.UpdateCacheOp: create_smart_golden_wrapper(
        update_cache_golden,
        convert_kwargs={"batch_offset": None},  # Filter out batch_offset parameter
    ),
=======
    # Elementwise unary operations
    ttir.GetDimensionSizeOp: torch.tensor,
    ttir.AbsOp: torch.abs,
    ttir.CeilOp: torch.ceil,
    ttir.CosOp: torch.cos,
    ttir.FloorOp: torch.floor,
    ttir.GeluOp: torch.nn.functional.gelu,
    ttir.IsFiniteOp: torch.isfinite,
    ttir.NegOp: torch.neg,
    ttir.TanOp: torch.tan,
    ttir.AtanOp: torch.atan,
    ttir.TanhOp: torch.tanh,
    ttir.ReciprocalOp: torch.reciprocal,
    ttir.ReluOp: torch.relu,
    ttir.RsqrtOp: torch.rsqrt,
    ttir.SigmoidOp: torch.sigmoid,
    ttir.SignOp: torch.sign,
    ttir.SinOp: torch.sin,
    ttir.SqrtOp: torch.sqrt,
    ttir.LogOp: torch.log,
    ttir.Log1pOp: torch.log1p,
    ttir.Expm1Op: torch.expm1,
    ttir.ExpOp: torch.exp,
    # Elementwise binary operations
    ttir.AddOp: torch.add,
    ttir.MultiplyOp: torch.multiply,
    ttir.SubtractOp: torch.subtract,
    ttir.DivOp: torch.div,
    ttir.MaximumOp: torch.maximum,
    ttir.MinimumOp: torch.minimum,
    ttir.RemainderOp: torch.remainder,
    ttir.PowOp: torch.pow,
    # Comparison operations
    ttir.EqualOp: torch.eq,
    ttir.NotEqualOp: torch.ne,
    ttir.GreaterEqualOp: torch.ge,
    ttir.GreaterThanOp: torch.gt,
    ttir.LessEqualOp: torch.le,
    ttir.LessThanOp: torch.lt,
    # Logical operations
    ttir.LogicalAndOp: torch.logical_and,
    ttir.LogicalOrOp: torch.logical_or,
    ttir.LogicalXorOp: torch.logical_xor,
    ttir.LogicalNotOp: torch.logical_not,
    # Selection operations
    ttir.WhereOp: torch.where,
    # Bitwise operations
    ttir.BitwiseAndOp: torch.bitwise_and,
    ttir.BitwiseOrOp: torch.bitwise_or,
    ttir.BitwiseXorOp: torch.bitwise_xor,
    ttir.BitwiseNotOp: torch.bitwise_not,
    # Reduction operations
    ttir.SumOp: torch.sum,
    ttir.MeanOp: torch.mean,
    ttir.MaxOp: max_golden,
    ttir.MinOp: torch.min,
    ttir.ProdOp: prod_golden,
    ttir.ReduceAndOp: torch.all,
    ttir.ReduceOrOp: torch.any,
    # Tensor manipulation
    ttir.TransposeOp: torch.transpose,
    ttir.ConcatOp: torch.concat,
    ttir.RepeatOp: torch.Tensor.repeat,
    ttir.RepeatInterleaveOp: torch.repeat_interleave,
    ttir.ReshapeOp: torch.reshape,
    ttir.SqueezeOp: torch.squeeze,
    ttir.UnsqueezeOp: torch.unsqueeze,
    ttir.ReverseOp: torch.flip,
    ttir.PermuteOp: torch.permute,
    ttir.ClampScalarOp: torch.clamp,
    ttir.ClampTensorOp: torch.clamp,
    ttir.BroadcastOp: torch.broadcast_to,
    ttir.PadOp: pad_golden,
    ttir.IndexSelectOp: select_golden,
    ttir.IndexOp: index_golden,
    ttir.SliceOp: slice_golden,
    ttir.GatherOp: gather_golden,
    # Neural network operations
    ttir.SoftmaxOp: torch.nn.functional.softmax,
    ttir.MatmulOp: torch.matmul,
    ttir.EmbeddingOp: embedding_golden,
    ttir.CumSumOp: torch.cumsum,
    ttir.Upsample2dOp: upsample2d_golden,
    # Type operations
    ttir.TypecastOp: torch.Tensor.type,
    # Tensor creation
    ttir.ZerosOp: torch.zeros,
    ttir.OnesOp: torch.ones,
    ttir.ArangeOp: arange_golden,
    # Quantization operations
    ttir.QuantizeOp: quantize_golden,
    ttir.DequantizeOp: torch.dequantize,
    ttir.RequantizeOp: requantize_golden,
    # Complex operations
    ttir.CbrtOp: cbrt_golden,
    ttir.Conv2dOp: conv2d_golden,
    ttir.MaxPool2dOp: max_pool2d_golden,
    ttir.ArgMaxOp: argmax_golden,
    ttir.LinearOp: linear_golden,
    ttir.DotGeneralOp: dot_general_golden,
    # Layout operations (identity functions)
    ttir.ToLayoutOp: lambda x: x,
    ttir.ViewLayoutOp: lambda x: x,
    # Cache operations
    ttir.FillCacheOp: fill_cache_golden,
    ttir.UpdateCacheOp: update_cache_golden,
>>>>>>> 1e49ed63c (all golden functions extracted into ttir_golden and ops.py refactored)
    # CCL (Collective Communication Library) operations
    ttir.MeshShardOp: mesh_shard_golden,
    ttir.AllGatherOp: all_gather_golden,
    ttir.AllReduceOp: all_reduce_golden,
    ttir.ReduceScatterOp: reduce_scatter_golden,
    ttir.CollectivePermuteOp: collective_permute_golden,
    # Operations with parameter transformations
<<<<<<< HEAD
    ttir.LeakyReluOp: create_smart_golden_wrapper(
        torch.nn.functional.leaky_relu, convert_kwargs={"parameter": "negative_slope"}
    ),
=======
    ttir.LeakyReluOp: torch.nn.functional.leaky_relu,
>>>>>>> 1e49ed63c (all golden functions extracted into ttir_golden and ops.py refactored)
}


# CCL (Collective Communication Library) Golden Functions
# We cannot inspect the intermediate buffer on a multi-device.
# Therefore, we only support Graph Level golden.
# Although generating an Op level golden is not needed,
# we return a random torch.Tensor with the correct output shape and type for TTIR.


def get_golden_function(ttir_op_class: type, **kwargs) -> Optional[Callable]:
    """
    @brief Get the golden function for a given TTIR operation class.
    @param ttir_op_class The TTIR operation class (e.g., ttir.AbsOp)
    @param kwargs Additional keyword arguments for specialized operation selection
    @return The corresponding golden function, or None if not found
    """
    # Handle special cases with parameters
    if ttir_op_class == ttir.ToLayoutOp and "tilize" in kwargs:
        if kwargs["tilize"]:
            return tilize_golden
        else:
            return untilize_golden

    return GOLDEN_MAPPINGS[ttir_op_class]
