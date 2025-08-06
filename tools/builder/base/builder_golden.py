# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Golden function mappings for TTIR and StableHLO operations.

This module provides a centralized mapping between TTIR and StableHLO operations and their
corresponding PyTorch golden reference implementations. Each golden function
serves as a reference implementation that produces the expected output for
comparison with TTIR or StableHLO operation results.
"""

from typing import Dict, Callable, Any, Optional, Union, List, Tuple
import torch
import torch.nn.functional
from ttmlir.dialects import ttir, stablehlo
from ttmlir.ir import Attribute


def cbrt_golden(x):
    """
    Custom golden function for cubic root.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor

    Returns
    -------
    torch.Tensor
        Tensor containing the cubic root of each element in the input tensor
    """
    golden_sign = torch.sign(x)
    golden_cbrt = torch.pow(torch.abs(x), 1 / 3)
    return torch.mul(golden_sign, golden_cbrt)


def conv2d_golden(
    input_tensor, weight, bias=None, stride=1, padding=0, dilation=1, groups=1
):
    """
    Custom golden function for conv2d with layout transformation.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Input tensor for convolution
    weight : torch.Tensor
        Convolution weight tensor
    bias : torch.Tensor, optional
        Optional bias tensor (default: None)
    stride : int, optional
        Stride for convolution (default: 1)
    padding : int, optional
        Padding for convolution (default: 0)
    dilation : int, optional
        Dilation for convolution (default: 1)
    groups : int, optional
        Number of groups for grouped convolution (default: 1)

    Returns
    -------
    torch.Tensor
        Result of 2D convolution with layout transformation
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


def conv_transpose2d_golden(
    input_tensor, weight, stride, padding, output_padding, dilation, groups
):
    """
    Custom golden function for conv_transpose2d with layout transformation.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Input tensor for transposed convolution
    weight : torch.Tensor
        Convolution weight tensor
    stride : Union[int, List[int]]
        Stride for transposed convolution
    padding : Union[int, List[int]]
        Padding for transposed convolution
    output_padding : Union[int, List[int]]
        Additional size added to output shape
    dilation : Union[int, List[int]]
        Dilation of the kernel
    groups : int
        Number of blocked connections from input to output channels

    Returns
    -------
    torch.Tensor
        Result of 2D transposed convolution with layout transformation
    """
    # Reorganize ttir_kwargs into golden_kwargs
    stride = list(stride) if not isinstance(stride, int) else int(stride)
    padding = list(padding) if not isinstance(padding, int) else int(padding)
    output_padding = (
        list(output_padding)
        if not isinstance(output_padding, int)
        else int(output_padding)
    )
    dilation = list(dilation) if not isinstance(dilation, int) else int(dilation)
    golden_bias = torch.rand((weight.size()[0]), dtype=input_tensor.dtype)

    # Reorganize input and output tensors, golden and ttir functions have different expected tensor shapes
    input_tensor = input_tensor.transpose(-2, -1).transpose(-3, -2)
    result = torch.nn.functional.conv_transpose2d(
        input_tensor,
        weight,
        bias=golden_bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
    )
    result = result.transpose(-3, -2).transpose(-2, -1)
    return result


def max_pool2d_golden(input_tensor, kernel_size, stride, padding, dilation, ceil_mode):
    """
    Custom golden function for max_pool2d with layout transformation.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Input tensor for max pooling
    kernel_size : Union[int, List[int]]
        Size of the pooling kernel
    stride : Union[int, List[int]]
        Stride for pooling operation
    padding : Union[int, List[int]]
        Padding for pooling operation
    dilation : Union[int, List[int]]
        Dilation for pooling operation
    ceil_mode : bool
        Whether to use ceiling mode for pooling

    Returns
    -------
    torch.Tensor
        Result of 2D max pooling with layout transformation
    """
    # Convert padding from [top, left, bottom, right] format to PyTorch format
    if isinstance(padding, (list, tuple)) and len(padding) == 4:
        # PyTorch MaxPool2d expects symmetric padding: (height_padding, width_padding)
        top, left, bottom, right = padding
        # For symmetric padding, top should equal bottom and left should equal right
        if top == bottom and left == right:
            torch_padding = (top, left)
        else:
            # For asymmetric padding, we need to manually pad the input tensor first
            # and then use zero padding for the MaxPool2d operation
            import torch.nn.functional as F

            # PyTorch F.pad expects padding in reverse order: [left, right, top, bottom]
            manual_padding = [left, right, top, bottom]
            input_tensor = F.pad(
                input_tensor, manual_padding, mode="constant", value=float("-inf")
            )
            torch_padding = 0
    else:
        torch_padding = padding

    # TTIR max_pool2d is channels last. PyTorch max_pool2d is channels first.
    maxpool_object = torch.nn.MaxPool2d(
        kernel_size, stride, torch_padding, dilation, ceil_mode
    )
    input_tensor = input_tensor.transpose(-2, -1).transpose(-3, -2)
    result = maxpool_object(input_tensor)
    result = result.transpose(-3, -2).transpose(-2, -1)
    return result


def batch_norm_golden(
    input_tensor,
    scale,
    offset,
    mean,
    variance,
    epsilon: float = 1e-5,
    training: bool = False,
    dim: int = 1,
):
    """
    Custom golden function for batch normalization with layout transformation.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Input tensor for batch normalization
    scale : torch.Tensor
        Scale tensor for batch normalization
    offset : torch.Tensor
        Offset tensor for batch normalization
    mean : torch.Tensor
        Mean tensor for batch normalization
    variance : torch.Tensor
        Variance tensor for batch normalization
    epsilon : float, optional
        Small value to avoid division by zero (default: 1e-5)
    training : bool, optional
        Whether the model is in training mode (default: False)
    dim : int, optional
        Dimension to apply batch normalization over (default: 1)

    Returns
    -------
    torch.Tensor
        Result of batch normalization with layout transformation
    """
    perm = list(range(input_tensor.ndim))
    perm[1], perm[dim] = perm[dim], perm[1]
    input_tensor = input_tensor.permute(perm)
    result = torch.nn.functional.batch_norm(
        input_tensor,
        running_mean=mean,
        running_var=variance,
        weight=scale,
        bias=offset,
        training=training,
        eps=epsilon,
    )
    inv_perm = [perm.index(i) for i in range(len(perm))]
    result = result.permute(inv_perm)
    return result


def argmax_golden(input_tensor, dim_arg, keep_dim=False):
    """
    Custom golden function for argmax.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Input tensor to find argmax of
    dim_arg : List[int]
        List containing dimension to find argmax along
    keep_dim : bool, optional
        Whether to keep the reduced dimension (default: False)

    Returns
    -------
    torch.Tensor
        Indices of maximum values along specified dimension as int32 tensor
    """
    result = torch.argmax(input_tensor, dim=dim_arg[0], keepdim=keep_dim)
    return result.to(torch.int32)


def linear_golden(a, b, bias=None, transpose_a=False, transpose_b=False):
    """
    Custom golden function for linear transformation.

    Parameters
    ----------
    a : torch.Tensor
        First input tensor
    b : torch.Tensor
        Second input tensor
    bias : torch.Tensor, optional
        Optional bias tensor (default: None)
    transpose_a : bool, optional
        Whether to transpose tensor a (default: False)
    transpose_b : bool, optional
        Whether to transpose tensor b (default: False)

    Returns
    -------
    torch.Tensor
        Result of linear transformation with optional bias
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
    Custom golden function for dot_general operation.

    Parameters
    ----------
    lhs : torch.Tensor
        Left-hand side tensor
    rhs : torch.Tensor
        Right-hand side tensor
    out : torch.Tensor
        Output tensor shape reference
    batch_dims_lhs : List[int]
        Batch dimensions for left tensor
    contract_dims_lhs : List[int]
        Contraction dimensions for left tensor
    batch_dims_rhs : List[int]
        Batch dimensions for right tensor
    contract_dims_rhs : List[int]
        Contraction dimensions for right tensor

    Returns
    -------
    torch.Tensor
        Result of generalized dot product operation
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
    Custom golden function for quantize operation.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Input tensor to quantize
    scale : float
        Scale factor for quantization
    zero_point : int
        Zero point for quantization
    dtype : torch.dtype
        Target quantized data type

    Returns
    -------
    torch.Tensor
        Quantized tensor as integer representation
    """
    return torch.quantize_per_tensor(input_tensor, scale, zero_point, dtype).int_repr()


def requantize_golden(input_tensor, scale, zero_point, dtype):
    """
    Custom golden function for requantize operation.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Input quantized tensor to requantize
    scale : float
        Scale factor for requantization
    zero_point : int
        Zero point for requantization
    dtype : torch.dtype
        Target quantized data type

    Returns
    -------
    torch.Tensor
        Requantized tensor
    """
    return torch.quantize_per_tensor(
        torch.dequantize(input_tensor), scale, zero_point, dtype
    )


def max_golden(input_tensor, dim_arg=None, keep_dim=True):
    """
    Custom golden function for max operation with conditional logic.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Input tensor to find maximum of
    dim_arg : int, optional
        Dimension to find maximum along (default: None for all dimensions)
    keep_dim : bool, optional
        Whether to keep the reduced dimension (default: True)

    Returns
    -------
    torch.Tensor
        Maximum values along specified dimension or global maximum
    """
    if dim_arg is not None:
        return torch.max(input_tensor, dim=dim_arg, keepdim=keep_dim)
    else:
        # For all dimensions reduction, reshape to match expected output
        result = torch.max(input_tensor)
        output_shape = [1] * input_tensor.dim()
        return result.reshape(*output_shape)


def prod_golden(input_tensor, dim_arg, keep_dim=False):
    """
    Custom golden function for prod operation with conditional logic.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Input tensor to compute product of
    dim_arg : List[int]
        List of dimensions to compute product along
    keep_dim : bool, optional
        Whether to keep the reduced dimension (default: False)

    Returns
    -------
    torch.Tensor
        Product of tensor elements along specified dimensions
    """
    if len(dim_arg) == 1:
        return torch.prod(input_tensor, dim=dim_arg[0], keepdim=keep_dim)
    else:
        # Multiple dimensions - reduce to scalar
        return torch.tensor([torch.prod(input_tensor).item()])


def embedding_golden(indices_tensor, weight_tensor):
    """
    Custom golden function for embedding operation.

    Parameters
    ----------
    indices_tensor : torch.Tensor
        Tensor containing indices to look up
    weight_tensor : torch.Tensor
        Weight tensor containing embedding vectors

    Returns
    -------
    torch.Tensor
        Embedded vectors corresponding to input indices
    """
    embedding = torch.nn.Embedding.from_pretrained(weight_tensor)
    golden_typecast = indices_tensor.to(torch.int32)
    golden_input = torch.clamp(golden_typecast, 0, (weight_tensor.size()[0] - 1))
    return embedding(golden_input)


def pad_golden(input_tensor, padding, value):
    """
    Custom golden function for pad operation with dimension reformatting.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Input tensor to pad
    padding : List[int]
        Padding specification
    value : Union[int, float]
        Value to use for padding

    Returns
    -------
    torch.Tensor
        Padded tensor
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
    Custom golden function for select operation.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Input tensor to select from
    dim : int
        Dimension to select along
    begin : int
        Starting index for selection
    length : int
        Length of selection
    stride : int
        Stride for selection

    Returns
    -------
    torch.Tensor
        Selected tensor slice
    """
    end = begin + length - 1
    index = torch.tensor([begin, end])
    return torch.index_select(input_tensor, dim=dim, index=index)


def index_golden(input_tensor, dim, begin, end, step):
    """
    Custom golden function for index operation.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Input tensor to index
    dim : int
        Dimension to index along
    begin : int
        Starting index
    end : int
        Ending index
    step : int
        Step size for indexing

    Returns
    -------
    torch.Tensor
        Indexed tensor
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
    Custom golden function for arange operation.

    Parameters
    ----------
    single_dim_tensor : torch.Tensor
        Single dimension tensor specification
    repeats : int
        Number of repeats for the range

    Returns
    -------
    torch.Tensor
        Generated range tensor
    """
    return single_dim_tensor.repeat(repeats)


def slice_golden(input_tensor, begins, ends, step):
    """
    Custom golden function for slice operation.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Input tensor to slice
    begins : List[int]
        Starting indices for each dimension
    ends : List[int]
        Ending indices for each dimension
    step : List[int]
        Step sizes for each dimension

    Returns
    -------
    torch.Tensor
        Sliced tensor
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
    Custom golden function for gather operation.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Input tensor to gather from
    start_indices_tensor : torch.Tensor
        Tensor containing starting indices
    offset_dims : List[int]
        Offset dimensions for gathering
    collapsed_slice_dims : List[int]
        Dimensions to collapse after slicing
    operand_batching_dims : List[int]
        Batching dimensions for operand
    start_indices_batching_dims : List[int]
        Batching dimensions for start indices
    start_index_map : List[int]
        Mapping of start indices
    index_vector_dim : int
        Dimension containing index vectors
    slice_sizes : List[int]
        Sizes of slices to gather
    indices_are_sorted : bool, optional
        Whether indices are sorted (default: False)

    Returns
    -------
    torch.Tensor
        Gathered tensor
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


def tilize_golden(input_tensor, tilize=True):
    """
    Custom golden function for tilize operation.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Input tensor to tilize
    tilize : bool, optional
        Tilize parameter (ignored, for compatibility) (default: True)

    Returns
    -------
    torch.Tensor
        Tilized tensor with proper tile layout transformation
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


def untilize_golden(input_tensor, tilize=False):
    """
    Custom golden function for untilize operation.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Input tensor to untilize
    tilize : bool, optional
        Tilize parameter (ignored, for compatibility) (default: False)

    Returns
    -------
    torch.Tensor
        Untilized tensor with proper layout transformation
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
    Custom golden function for upsample2d operation.

    Parameters
    ----------
    in0 : torch.Tensor
        Input tensor to upsample
    in1 : torch.Tensor
        Output tensor specification
    scale_factor : Union[int, List[int]]
        Scaling factor for upsampling
    mode : str, optional
        Upsampling mode (default: "nearest")

    Returns
    -------
    torch.Tensor
        Upsampled 2D tensor
    """
    transposed_golden = torch.transpose(in0, 1, 3)
    golden_output_shape = in1.shape[1:-1]
    output = torch.nn.functional.interpolate(
        transposed_golden, size=golden_output_shape, mode=mode
    )
    return torch.transpose(output, 1, 3)


def fill_cache_golden(cache_tensor, input_tensor):
    """
    Custom golden function for fill_cache operation.

    Parameters
    ----------
    cache_tensor : torch.Tensor
        Cache tensor to fill
    input_tensor : torch.Tensor
        Input tensor data

    Returns
    -------
    torch.Tensor
        Filled cache tensor
    """
    result = cache_tensor.clone()
    result[:, :, : input_tensor.shape[2], :] = input_tensor
    return result


def update_cache_golden(cache_tensor, update_tensor, indices_tensor):
    """
    Custom golden function for update_cache operation.

    Parameters
    ----------
    cache_tensor : torch.Tensor
        Cache tensor to update
    update_tensor : torch.Tensor
        Tensor containing update data
    indices_tensor : torch.Tensor
        Tensor containing update indices

    Returns
    -------
    torch.Tensor
        Updated cache tensor
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
    Return a random torch.Tensor which has the correct shape and type after doing mesh_shard on the input.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor to be sharded
    mesh_shape : Tuple[int, int]
        Shape of the device mesh
    shard_type : Attribute
        Type of sharding operation
    shard_direction : Attribute
        Direction of sharding
    shard_shape : Tuple[int, int]
        Shape of the shard
    shard_dims : List[int]
        Dimensions to shard along

    Returns
    -------
    torch.Tensor
        Random tensor with correct output shape and type
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
    Return a random torch.Tensor which has the correct shape and type after doing all_gather on the input.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor to gather from all devices
    mesh_shape : Tuple[int, int]
        Shape of the device mesh
    all_gather_dim : int
        Dimension to gather along
    cluster_axis : int
        Axis of the cluster for gathering

    Returns
    -------
    torch.Tensor
        Random tensor with correct output shape and type
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
    Return a random torch.Tensor which has the correct shape and type after doing all_reduce on the input.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor to reduce across devices
    mesh_shape : Tuple[int, int]
        Shape of the device mesh
    cluster_axis : int
        Axis of the cluster for reduction
    reduce_type : Attribute
        Type of reduction operation

    Returns
    -------
    torch.Tensor
        Random tensor with correct output shape and type
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
    Return a random torch.Tensor which has the correct shape and type after doing reduce_scatter on the input.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor to reduce and scatter
    mesh_shape : Tuple[int, int]
        Shape of the device mesh
    reduce_type : Attribute
        Type of reduction operation
    scatter_dim : int
        Dimension to scatter along
    cluster_axis : int
        Axis of the cluster for operation

    Returns
    -------
    torch.Tensor
        Random tensor with correct output shape and type
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
    Return a random torch.Tensor which has the correct shape and type after doing collective_permute on the input.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor to permute across devices
    mesh_shape : Tuple[int, int]
        Shape of the device mesh
    source_target_pairs : List[Tuple[int, int]]
        List of (source, target) device ID pairs for permutation

    Returns
    -------
    torch.Tensor
        Random tensor with correct output shape and type
    """
    return torch.randn(input.shape, dtype=input.dtype)


def all_to_all_golden(
    input: torch.Tensor,
    mesh_shape: Tuple[int, int],
    split_dim: int,
    concat_dim: int,
    split_count: int,
    replica_groups: List[List[int]],
) -> torch.Tensor:
    """
    Return a random torch.Tensor which has the correct shape and type after doing all_to_all on the input.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor to perform all-to-all communication on
    mesh_shape : Tuple[int, int]
        Shape of the device mesh
    split_dim : int
        Dimension to split the input tensor along
    concat_dim : int
        Dimension to concatenate the received tensors along
    split_count : int
        Number of splits to perform
    replica_groups : List[List[int]]
        Groups of replica devices for communication

    Returns
    -------
    torch.Tensor
        Random tensor with correct output shape and type
    """
    out_shape = list(input.shape)
    out_shape[split_dim] //= split_count
    out_shape[concat_dim] *= split_count
    return torch.randn(out_shape, dtype=input.dtype)


def collective_broadcast_golden(
    input: torch.Tensor,
    mesh_shape: Tuple[int, int],
    replica_groups: List[Tuple[int, int]],
) -> torch.Tensor:
    """
    Return a random torch.Tensor which has the correct shape and type after doing collective_broadcast on the input.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor to broadcast across devices
    mesh_shape : Tuple[int, int]
        Shape of the device mesh
    replica_groups : List[Tuple[int, int]]
        Groups of replica devices for broadcasting

    Returns
    -------
    torch.Tensor
        Random tensor with correct output shape and type
    """
    return torch.randn(input.shape, dtype=input.dtype)


"""
Dictionary mapping TTIR operation classes to their corresponding golden functions.

This dictionary provides a centralized mapping between TTIR operation types and their
PyTorch-based golden reference implementations. Each key is a TTIR operation class
(e.g., ttir.AbsOp) and each value is the corresponding golden function that computes
the expected output for that operation.

The mapping supports:
    - Elementwise unary operations (abs, ceil, cos, etc.)
    - Elementwise binary operations (add, multiply, subtract, etc.)
    - Elementwise ternary operations (where, select, etc.)
    - Comparison operations (eq, ne, lt, gt, etc.)
    - Bitwise operations (and, or, xor, not)
    - Reduction operations (sum, mean, max, min, etc.)
    - Tensor manipulation (transpose, concat, reshape, etc.)
    - Neural network operations (matmul, embedding, conv2d, etc.)
    - Layout operations (to_layout, view_layout)
    - Quantization operations (quantize, dequantize, requantize)
    - Collective communication operations (all_gather, all_reduce, etc.)

Usage:
    golden_fn = GOLDEN_MAPPINGS.get(ttir.AbsOp)
    if golden_fn:
        result = golden_fn(input_tensor)
"""
GOLDEN_MAPPINGS: Dict[type, Callable] = {
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
    ttir.BatchNormOp: batch_norm_golden,
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
    ttir.ConvTranspose2dOp: conv_transpose2d_golden,
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
    # CCL (Collective Communication Library) operations
    ttir.MeshShardOp: mesh_shard_golden,
    ttir.AllGatherOp: all_gather_golden,
    ttir.AllReduceOp: all_reduce_golden,
    ttir.ReduceScatterOp: reduce_scatter_golden,
    ttir.CollectivePermuteOp: collective_permute_golden,
    ttir.AllToAllOp: all_to_all_golden,
    ttir.CollectiveBroadcastOp: collective_broadcast_golden,
    # Operations with parameter transformations
    ttir.LeakyReluOp: torch.nn.functional.leaky_relu,
    stablehlo.AddOp: torch.add,
}


# CCL (Collective Communication Library) Golden Functions
# We cannot inspect the intermediate buffer on a multi-device.
# Therefore, we only support Graph Level golden.
# Although generating an Op level golden is not needed,
# we return a random torch.Tensor with the correct output shape and type for TTIR.


def get_golden_function(ttir_op_class: type, **kwargs) -> Optional[Callable]:
    """
    Get the golden function for a given TTIR operation class.

    Parameters
    ----------
    ttir_op_class : type
        The TTIR operation class (e.g., ttir.AbsOp)
    **kwargs
        Additional keyword arguments for specialized operation selection

    Returns
    -------
    Optional[Callable]
        The corresponding golden function, or None if not found
    """
    # Handle special cases with parameters
    if ttir_op_class == ttir.ToLayoutOp and "tilize" in kwargs:
        if kwargs["tilize"]:
            return tilize_golden
        else:
            return untilize_golden

    return GOLDEN_MAPPINGS[ttir_op_class]
