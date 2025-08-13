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
from ttmlir.ir import (
    Attribute,
    ArrayAttr,
    IntegerAttr,
    IntegerType,
    BoolAttr,
    DenseI32ArrayAttr,
    DenseI64ArrayAttr,
)


def unpack_mlir_attr(attr):
    """Unpack MLIR attributes into plain Python values.

    Supports IntegerAttr, BoolAttr, DenseI32ArrayAttr, DenseI64ArrayAttr, ArrayAttr,
    as well as native Python list/tuple/int/bool. Raises ValueError for unsupported types.
    """
    if isinstance(attr, IntegerAttr):
        return attr.value
    if isinstance(attr, BoolAttr):
        return attr.value
    if isinstance(attr, (DenseI64ArrayAttr, DenseI32ArrayAttr)):
        return list(attr)
    if isinstance(attr, ArrayAttr):
        return [unpack_mlir_attr(item) for item in attr]
    if isinstance(attr, (list, tuple)):
        return list(attr)
    if isinstance(attr, (int, bool)):
        return attr
    raise ValueError(f"Unexpected attribute type: {type(attr)}")


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
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
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
    **kwargs : dict
        Keyword arguments containing:
        - stride: Union[int, List[int]] - Stride for convolution (default: 1)
        - padding: Union[int, List[int]] - Padding for convolution (default: 0)
        - dilation: Union[int, List[int]] - Dilation for convolution (default: 1)
        - groups: int - Number of groups for grouped convolution (default: 1)

    Returns
    -------
    torch.Tensor
        Result of 2D convolution with layout transformation
    """
    # ttir can handle a broadcastable bias in the shape [1, 1, 1, C_out], but PyTorch requires the bias to be rank 1: [C_out].
    if bias is not None:
        bias = bias.squeeze()  # Removes all dims of size 1

    # Get parameters from ttir_kwargs
    stride = kwargs.get("stride", 1)
    padding = kwargs.get("padding", 0)
    dilation = kwargs.get("dilation", 1)
    groups = kwargs.get("groups", 1)

    stride = unpack_mlir_attr(stride)
    padding = unpack_mlir_attr(padding)
    dilation = unpack_mlir_attr(dilation)

    # Reorganize input and output tensors, golden and ttir functions have different expected tensor shapes
    input_tensor = input_tensor.transpose(-2, -1).transpose(-3, -2)

    if input_tensor.is_quantized:
        if not weight.is_quantized:
            raise ValueError("Quantized input requires quantized weight.")
        # if input tensor and weight tensor zero points are different, error out
        if (input_tensor.q_zero_point() - 128) != weight.q_zero_point():
            raise ValueError("Input and weight zero points must be the same.")
        # Pack weights and bias for quantized conv.
        packed_weight = torch.ops.quantized.conv2d_prepack(
            weight,
            bias,
            stride=[stride] * 2 if isinstance(stride, int) else stride,
            padding=[padding] * 2 if isinstance(padding, int) else padding,
            dilation=[dilation] * 2 if isinstance(dilation, int) else dilation,
            groups=groups,
        )

        # Convert to int_repr to match the builder golden function.
        result = torch.ops.quantized.conv2d(
            input_tensor,
            packed_weight,
            input_tensor.q_scale() * weight.q_scale(),
            input_tensor.q_zero_point(),
        ).int_repr()

    else:
        if bias is not None:
            bias = bias.squeeze()

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
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Custom golden function for conv_transpose2d with layout transformation.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Input tensor for transposed convolution
    weight : torch.Tensor
        Convolution weight tensor
    bias : Optional[torch.Tensor]
        Optional bias tensor
    **kwargs : dict
        Keyword arguments containing:
        - stride: Union[int, List[int]] - Stride for transposed convolution
        - padding: Union[int, List[int]] - Padding for transposed convolution
        - output_padding: Union[int, List[int]] - Additional size added to output shape
        - dilation: Union[int, List[int]] - Dilation of the kernel
        - groups: int - Number of blocked connections from input to output channels

    Returns
    -------
    torch.Tensor
        Result of 2D transposed convolution with layout transformation
    """
    # Get parameters from ttir_kwargs
    stride = kwargs.get("stride", 1)
    padding = kwargs.get("padding", 0)
    output_padding = kwargs.get("output_padding", 0)
    dilation = kwargs.get("dilation", 1)
    groups = kwargs.get("groups", 1)

    stride = unpack_mlir_attr(stride)
    padding = unpack_mlir_attr(padding)
    output_padding = unpack_mlir_attr(output_padding)
    dilation = unpack_mlir_attr(dilation)
    groups = unpack_mlir_attr(groups)
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


def max_pool2d_golden(input_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Custom golden function for max_pool2d with layout transformation.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Input tensor for max pooling
    **kwargs : dict
        Keyword arguments containing:
        - kernel_size: Union[int, List[int]] - Size of the pooling kernel
        - stride: Union[int, List[int]] - Stride for pooling operation
        - padding: Union[int, List[int]] - Padding for pooling operation
        - dilation: Union[int, List[int]] - Dilation for pooling operation
        - ceil_mode: bool - Whether to use ceiling mode for pooling

    Returns
    -------
    torch.Tensor
        Result of 2D max pooling with layout transformation
    """
    # Get parameters from ttir_kwargs
    kernel_size = kwargs.get("kernel")
    stride = kwargs.get("stride", kernel_size)  # Default stride = kernel size
    padding = kwargs.get("padding", 0)
    dilation = kwargs.get("dilation", 1)
    ceil_mode = kwargs.get("ceil_mode", False)

    kernel_size = unpack_mlir_attr(kernel_size)
    stride = unpack_mlir_attr(stride)
    padding = unpack_mlir_attr(padding)
    dilation = unpack_mlir_attr(dilation)

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


def logical_not_golden(input_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Golden function for logical_not operation.

    Elementwise logical NOT.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Input tensor to invert logically.
    **kwargs : dict
        Keyword arguments (unused for this operation).

    Returns
    -------
    torch.Tensor
        Tensor with logical NOT of input_tensor, cast back to input dtype.
    """
    # Compute bool result then cast to match input dtype
    result_bool = torch.logical_not(input_tensor)
    return result_bool.to(input_tensor.dtype)


def equal_golden(
    input_tensor: torch.Tensor, other_tensor: torch.Tensor, **kwargs
) -> torch.Tensor:
    """
    Golden function for equal (eq) operation.

    Elementwise equality comparison.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Left-hand side tensor.
    other_tensor : torch.Tensor
        Right-hand side tensor.

    Returns
    -------
    torch.Tensor
        Tensor with the same dtype as input_tensor containing the equality results.
    """
    result_bool = torch.eq(input_tensor, other_tensor)
    return result_bool.to(input_tensor.dtype)


def not_equal_golden(
    input_tensor: torch.Tensor, other_tensor: torch.Tensor, **kwargs
) -> torch.Tensor:
    """
    Golden function for not_equal (ne) operation.

    Elementwise inequality comparison.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Left-hand side tensor.
    other_tensor : torch.Tensor
        Right-hand side tensor.

    Returns
    -------
    torch.Tensor
        Tensor with the same dtype as input_tensor containing the inequality results.
    """
    result_bool = torch.ne(input_tensor, other_tensor)
    return result_bool.to(input_tensor.dtype)


def greater_equal_golden(
    input_tensor: torch.Tensor, other_tensor: torch.Tensor, **kwargs
) -> torch.Tensor:
    """
    Golden function for greater_equal (ge) operation.

    Elementwise greater-than-or-equal comparison.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Left-hand side tensor.
    other_tensor : torch.Tensor
        Right-hand side tensor.

    Returns
    -------
    torch.Tensor
        Tensor with the same dtype as input_tensor containing the comparison results.
    """
    result_bool = torch.ge(input_tensor, other_tensor)
    return result_bool.to(input_tensor.dtype)


def greater_than_golden(
    input_tensor: torch.Tensor, other_tensor: torch.Tensor, **kwargs
) -> torch.Tensor:
    """
    Golden function for greater_than (gt) operation.

    Elementwise greater-than comparison.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Left-hand side tensor.
    other_tensor : torch.Tensor
        Right-hand side tensor.

    Returns
    -------
    torch.Tensor
        Tensor with the same dtype as input_tensor containing the comparison results.
    """
    result_bool = torch.gt(input_tensor, other_tensor)
    return result_bool.to(input_tensor.dtype)


def less_equal_golden(
    input_tensor: torch.Tensor, other_tensor: torch.Tensor, **kwargs
) -> torch.Tensor:
    """
    Golden function for less_equal (le) operation.

    Elementwise less-than-or-equal comparison.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Left-hand side tensor.
    other_tensor : torch.Tensor
        Right-hand side tensor.

    Returns
    -------
    torch.Tensor
        Tensor with the same dtype as input_tensor containing the comparison results.
    """
    result_bool = torch.le(input_tensor, other_tensor)
    return result_bool.to(input_tensor.dtype)


def less_than_golden(
    input_tensor: torch.Tensor, other_tensor: torch.Tensor, **kwargs
) -> torch.Tensor:
    """
    Golden function for less_than (lt) operation.

    Elementwise less-than comparison.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Left-hand side tensor.
    other_tensor : torch.Tensor
        Right-hand side tensor.

    Returns
    -------
    torch.Tensor
        Tensor with the same dtype as input_tensor containing the comparison results.
    """
    result_bool = torch.lt(input_tensor, other_tensor)
    return result_bool.to(input_tensor.dtype)


def logical_and_golden(
    input_tensor: torch.Tensor, other_tensor: torch.Tensor, **kwargs
) -> torch.Tensor:
    """
    Golden function for logical_and operation.

    Elementwise logical AND.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Left-hand side tensor.
    other_tensor : torch.Tensor
        Right-hand side tensor.

    Returns
    -------
    torch.Tensor
        Tensor with the same dtype as input_tensor containing the logical AND results.
    """
    result_bool = torch.logical_and(input_tensor, other_tensor)
    return result_bool.to(input_tensor.dtype)


def logical_or_golden(
    input_tensor: torch.Tensor, other_tensor: torch.Tensor, **kwargs
) -> torch.Tensor:
    """
    Golden function for logical_or operation.

    Elementwise logical OR.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Left-hand side tensor.
    other_tensor : torch.Tensor
        Right-hand side tensor.

    Returns
    -------
    torch.Tensor
        Tensor with the same dtype as input_tensor containing the logical OR results.
    """
    result_bool = torch.logical_or(input_tensor, other_tensor)
    return result_bool.to(input_tensor.dtype)


def logical_xor_golden(
    input_tensor: torch.Tensor, other_tensor: torch.Tensor, **kwargs
) -> torch.Tensor:
    """
    Golden function for logical_xor operation.

    Elementwise logical XOR.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Left-hand side tensor.
    other_tensor : torch.Tensor
        Right-hand side tensor.

    Returns
    -------
    torch.Tensor
        Tensor with the same dtype as input_tensor containing the logical XOR results.
    """
    result_bool = torch.logical_xor(input_tensor, other_tensor)
    return result_bool.to(input_tensor.dtype)


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


def min_golden(input_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Golden function for min operation.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Input tensor
    **kwargs : dict
        Keyword arguments containing:
        - dim_arg: int, optional - Dimension to reduce over (default: None, reduces over all dimensions)
        - keep_dim: bool, optional - If True, retains reduced dimensions with length 1 (default: True)

    Returns
    -------
    torch.Tensor
        Tensor with minimum values
    """
    dim_arg = kwargs.get("dim_arg", None)
    keep_dim = kwargs.get("keep_dim", True)

    if dim_arg is None:
        return torch.min(input_tensor)
    else:
        # Extract the first dimension if dim_arg is a list
        if isinstance(dim_arg, list) and len(dim_arg) > 0:
            dim_arg = dim_arg[0]
        return torch.min(input_tensor, dim=dim_arg, keepdim=keep_dim)


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


def gather_golden(
    input_tensor: torch.Tensor, start_indices_tensor: torch.Tensor, **kwargs
) -> torch.Tensor:
    """
    Golden function for gather operation with TTIR parameter names.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Input tensor to gather from
    start_indices_tensor : torch.Tensor
        Tensor containing starting indices
    **kwargs : dict
        Keyword arguments including gather attributes as MLIR attributes

    Returns
    -------
    torch.Tensor
        Gathered tensor
    """

    # Unpack MLIR attributes from kwargs
    offset_dims = unpack_mlir_attr(kwargs.get("offset_dims", []))
    collapsed_slice_dims = unpack_mlir_attr(kwargs.get("collapsed_slice_dims", []))
    operand_batching_dims = unpack_mlir_attr(kwargs.get("operand_batching_dims", []))
    start_indices_batching_dims = unpack_mlir_attr(
        kwargs.get("start_indices_batching_dims", [])
    )
    start_index_map = unpack_mlir_attr(kwargs.get("start_index_map", []))
    index_vector_dim = unpack_mlir_attr(kwargs.get("index_vector_dim", 0))
    slice_sizes = unpack_mlir_attr(kwargs.get("slice_sizes", []))
    indices_are_sorted = unpack_mlir_attr(kwargs.get("indices_are_sorted", False))
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


def tilize_golden(input_tensor, tilize=True, **kwargs):
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


def untilize_golden(input_tensor, tilize=False, **kwargs):
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


def fill_cache_golden(cache_tensor, input_tensor, **kwargs):
    """
    Custom golden function for fill_cache operation.

    Parameters
    ----------
    cache_tensor : torch.Tensor
        Cache tensor to fill
    input_tensor : torch.Tensor
        Input tensor data
    **kwargs : dict
        Additional keyword arguments (batch_offset is ignored)

    Returns
    -------
    torch.Tensor
        Filled cache tensor
    """
    result = cache_tensor.clone()
    result[:, :, : input_tensor.shape[2], :] = input_tensor
    return result


def update_cache_golden(cache_tensor, update_tensor, indices_tensor, **kwargs):
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
    **kwargs : dict
        Additional keyword arguments (batch_offset is ignored)

    Returns
    -------
    torch.Tensor
        Updated cache tensor
    """
    result = cache_tensor.clone()
    # Simple update logic - this would need to be refined based on actual requirements
    result[:, :, : update_tensor.shape[2], :] = update_tensor
    return result


def get_dimension_size_golden(input_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Golden function for get_dimension_size operation.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Input tensor to get dimension size from
    **kwargs : dict
        Keyword arguments including 'dimension'

    Returns
    -------
    torch.Tensor
        Tensor containing the size of the specified dimension as int32
    """
    dimension = kwargs.get("dimension", 0)
    size = input_tensor.size(dimension)
    return torch.tensor([size], dtype=torch.int32)


def sum_golden(input_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Golden function for sum operation with TTIR parameter names.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Input tensor to sum
    **kwargs : dict
        Keyword arguments containing:
        - dim_arg: List[int] - Dimensions to reduce over (default: [0])
        - keep_dim: bool - If True, retains reduced dimensions with length 1 (default: True)

    Returns
    -------
    torch.Tensor
        Summed tensor
    """
    # Get parameters from ttir_kwargs
    dim_arg = kwargs.get("dim_arg", [0])
    keep_dim = kwargs.get("keep_dim", True)
    # Convert to torch.sum format
    return torch.sum(input_tensor, dim=dim_arg, keepdim=keep_dim)


def mean_golden(input_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Golden function for mean operation with TTIR parameter names.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Input tensor to compute mean of
    **kwargs : dict
        Keyword arguments including 'dim_arg' and 'keep_dim'

    Returns
    -------
    torch.Tensor
        Mean tensor
    """
    dim_arg = kwargs.get("dim_arg", [0])
    keep_dim = kwargs.get("keep_dim", True)
    return torch.mean(input_tensor, dim=dim_arg, keepdim=keep_dim)


def reduce_and_golden(input_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Golden function for reduce_and operation with TTIR parameter names.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Input tensor to reduce
    **kwargs : dict
        Keyword arguments including 'dim_arg' and 'keep_dim'

    Returns
    -------
    torch.Tensor
        Reduced tensor
    """
    dim_arg = kwargs.get("dim_arg", [0])
    keep_dim = kwargs.get("keep_dim", True)
    return torch.all(input_tensor, dim=tuple(dim_arg), keepdim=keep_dim)


def reduce_or_golden(input_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Golden function for reduce_or operation with TTIR parameter names.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Input tensor to reduce
    **kwargs : dict
        Keyword arguments including 'dim_arg' and 'keep_dim'

    Returns
    -------
    torch.Tensor
        Reduced tensor
    """
    dim_arg = kwargs.get("dim_arg", [0])
    keep_dim = kwargs.get("keep_dim", True)
    return torch.any(input_tensor, dim=tuple(dim_arg), keepdim=keep_dim)


def transpose_golden(input_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Golden function for transpose operation with TTIR parameter names.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Input tensor
    **kwargs : dict
        Keyword arguments including 'dim0' and 'dim1'

    Returns
    -------
    torch.Tensor
        Transposed tensor
    """
    dim0 = kwargs.get("dim0", 0)
    dim1 = kwargs.get("dim1", 1)
    return torch.transpose(input_tensor, dim0, dim1)


def concat_golden(input_tensors: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Golden function for concat operation with TTIR parameter names.

    Parameters
    ----------
    input_tensors : torch.Tensor
        Input tensors (will be unpacked from tuple)
    **kwargs : dict
        Keyword arguments including 'dim'

    Returns
    -------
    torch.Tensor
        Concatenated tensor
    """
    dim = kwargs.get("dim", 0)
    if isinstance(input_tensors, tuple):
        return torch.concat(input_tensors, dim=dim)
    else:
        return torch.concat([input_tensors], dim=dim)


# Investigate how repeat works in torch
def repeat_golden(input_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Golden function for repeat operation with TTIR parameter names.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Input tensor
    **kwargs : dict
        Keyword arguments including 'repeat_dimensions'

    Returns
    -------
    torch.Tensor
        Repeated tensor
    """
    repeat_dimensions = kwargs.get("repeat_dimensions", [1])
    return torch.Tensor.repeat(input_tensor, repeats=repeat_dimensions)


def reshape_golden(input_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Golden function for reshape operation with TTIR parameter names.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Input tensor
    **kwargs : dict
        Keyword arguments including 'shape'

    Returns
    -------
    torch.Tensor
        Reshaped tensor
    """
    shape = kwargs.get("shape", input_tensor.shape)
    return torch.reshape(input_tensor, shape)


def squeeze_golden(input_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Golden function for squeeze operation with TTIR parameter names.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Input tensor
    **kwargs : dict
        Keyword arguments including 'dim'

    Returns
    -------
    torch.Tensor
        Squeezed tensor
    """
    dim = kwargs.get("dim", None)
    return torch.squeeze(input_tensor, dim=dim)


def unsqueeze_golden(input_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Golden function for unsqueeze operation with TTIR parameter names.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Input tensor
    **kwargs : dict
        Keyword arguments including 'dim'

    Returns
    -------
    torch.Tensor
        Unsqueezed tensor
    """
    dim = kwargs.get("dim", 0)
    return torch.unsqueeze(input_tensor, dim=dim)


def clamp_scalar_golden(input_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Golden function for clamp_scalar operation with TTIR parameter names.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Input tensor
    **kwargs : dict
        Keyword arguments including 'min' and 'max'

    Returns
    -------
    torch.Tensor
        Clamped tensor
    """
    min_val = kwargs.get("min", None)
    max_val = kwargs.get("max", None)
    return torch.clamp(input_tensor, min=min_val, max=max_val)


def permute_golden(input_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Golden function for permute operation with TTIR parameter names.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Input tensor
    **kwargs : dict
        Keyword arguments including 'permutation' as MLIR attribute

    Returns
    -------
    torch.Tensor
        Permuted tensor
    """

    permutation = kwargs.get("permutation", None)
    if permutation is None:
        return input_tensor

    permutation = unpack_mlir_attr(permutation)
    return torch.permute(input_tensor, tuple(permutation))


def leaky_relu_golden(input_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Golden function for leaky_relu operation with TTIR parameter names.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Input tensor
    **kwargs : dict
        Keyword arguments including 'parameter'

    Returns
    -------
    torch.Tensor
        Leaky ReLU output
    """
    parameter = kwargs.get("parameter", 0.01)
    return torch.nn.functional.leaky_relu(input_tensor, negative_slope=parameter)


def softmax_golden(input_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Golden function for softmax operation with TTIR parameter names.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Input tensor
    **kwargs : dict
        Keyword arguments including 'dimension'

    Returns
    -------
    torch.Tensor
        Softmax output
    """
    dimension = kwargs.get("dimension", 1)
    return torch.nn.functional.softmax(input_tensor, dim=dimension)


def index_golden(input_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Golden function for index operation with TTIR parameter names.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Input tensor
    **kwargs : dict
        Keyword arguments including 'dim', 'begin', 'end', 'step'

    Returns
    -------
    torch.Tensor
        Indexed tensor
    """
    dim = kwargs.get("dim", 0)
    begin = kwargs.get("begin", 0)
    end = kwargs.get("end", None)
    step = kwargs.get("step", 1)

    if end is None:
        end = input_tensor.size(dim)

    indices = torch.arange(begin, end, step, device=input_tensor.device)
    return torch.index_select(input_tensor, dim, indices)


def slice_golden(input_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Golden function for slice operation with TTIR parameter names.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Input tensor
    **kwargs : dict
        Keyword arguments including slice attributes as MLIR attributes

    Returns
    -------
    torch.Tensor
        Sliced tensor
    """

    # Unpack MLIR attributes from kwargs
    begins = unpack_mlir_attr(kwargs.get("begins", [0]))
    ends = unpack_mlir_attr(kwargs.get("ends", None))
    step = unpack_mlir_attr(kwargs.get("step", [1]))

    if ends is None:
        ends = [input_tensor.size(i) for i in range(len(begins))]

    # Build slice objects for each dimension
    slices = []
    for i in range(len(begins)):
        start = begins[i] if i < len(begins) else 0
        end = ends[i] if i < len(ends) else input_tensor.size(i)
        step_val = step[i] if i < len(step) else 1
        slices.append(slice(start, end, step_val))

    return input_tensor[slices]


def zeros_golden(**kwargs) -> torch.Tensor:
    """
    Golden function for zeros operation with TTIR parameter names.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments including 'size'

    Returns
    -------
    torch.Tensor
        Zero tensor
    """
    size = kwargs.get("shape", [1])
    return torch.zeros(size)


def ones_golden(**kwargs) -> torch.Tensor:
    """
    Golden function for ones operation with TTIR parameter names.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments including 'size'

    Returns
    -------
    torch.Tensor
        Ones tensor
    """
    size = kwargs.get("shape", [1])
    return torch.ones(size)


def reverse_golden(input_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Golden function for reverse operation with TTIR parameter names.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Input tensor
    **kwargs : dict
        Keyword arguments including 'dims'

    Returns
    -------
    torch.Tensor
        Reversed tensor
    """
    dims = kwargs.get("dimensions", [0])
    return torch.flip(input_tensor, dims)


def arange_golden(single_dim_tensor, repeats):
    """
    Golden function for arange operation using TTIR kwargs.

    Expected kwargs from builder (ttir_kwargs):
    - start: int
    - end: int
    - step: int
    - arange_dimension: int (ignored here; layout handled by builder output shape)
    """
    # start = kwargs.get("start", 0)
    # end = kwargs.get("end", 0)
    # step = kwargs.get("step", 1)
    # return torch.arange(start=start, end=end, step=step)
    return single_dim_tensor.repeat(repeats)


def cumsum_golden(input_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Golden function for cumsum operation with TTIR parameter names.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Input tensor
    **kwargs : dict
        Keyword arguments containing:
        - dim: int - Dimension along which to compute cumulative sum

    Returns
    -------
    torch.Tensor
        Cumulative sum of input tensor along specified dimension
    """
    dim = kwargs.get("dim", 0)  # Use the dim parameter from ttir_kwargs
    return torch.cumsum(input_tensor, dim=dim)


def repeat_interleave_golden(input_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
    """
    Golden function for repeat_interleave operation with TTIR parameter names.

    Parameters
    ----------
    input_tensor : torch.Tensor
        Input tensor
    **kwargs : dict
        Keyword arguments including 'repeats' and 'dim'

    Returns
    -------
    torch.Tensor
        Repeated tensor
    """
    repeats = kwargs.get("repeats", 1)
    dim = kwargs.get("dim", 0)
    return torch.repeat_interleave(input_tensor, repeats, dim=dim)


# CCL (Collective Communication Library) Golden Functions
# We cannot inspect the intermediate buffer on a multi-device.
# Therefore, we only support Graph Level golden.
# Although generating an Op level golden is not needed,
# we return a random torch.Tensor with the correct output shape and type for TTIR.


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
    ttir.GetDimensionSizeOp: get_dimension_size_golden,
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
    ttir.EqualOp: equal_golden,
    ttir.NotEqualOp: not_equal_golden,
    ttir.GreaterEqualOp: greater_equal_golden,
    ttir.GreaterThanOp: greater_than_golden,
    ttir.LessEqualOp: less_equal_golden,
    ttir.LessThanOp: less_than_golden,
    # Logical operations
    ttir.LogicalAndOp: logical_and_golden,
    ttir.LogicalOrOp: logical_or_golden,
    ttir.LogicalXorOp: logical_xor_golden,
    ttir.LogicalNotOp: logical_not_golden,
    # Selection operations
    ttir.WhereOp: torch.where,
    # Bitwise operations
    ttir.BitwiseAndOp: torch.bitwise_and,
    ttir.BitwiseOrOp: torch.bitwise_or,
    ttir.BitwiseXorOp: torch.bitwise_xor,
    ttir.BitwiseNotOp: torch.bitwise_not,
    # Reduction operations
    ttir.SumOp: sum_golden,
    ttir.MeanOp: mean_golden,
    ttir.MaxOp: max_golden,
    ttir.MinOp: min_golden,
    ttir.ProdOp: prod_golden,
    ttir.ReduceAndOp: reduce_and_golden,
    ttir.ReduceOrOp: reduce_or_golden,
    # Tensor manipulation
    ttir.TransposeOp: transpose_golden,
    ttir.ConcatOp: concat_golden,
    ttir.RepeatOp: repeat_golden,
    ttir.RepeatInterleaveOp: repeat_interleave_golden,
    ttir.ReshapeOp: reshape_golden,
    ttir.SqueezeOp: squeeze_golden,
    ttir.UnsqueezeOp: unsqueeze_golden,
    ttir.ReverseOp: reverse_golden,
    ttir.PermuteOp: permute_golden,
    ttir.ClampScalarOp: clamp_scalar_golden,
    ttir.ClampTensorOp: torch.clamp,
    ttir.CumSumOp: cumsum_golden,
    ttir.BroadcastOp: torch.broadcast_to,
    ttir.PadOp: pad_golden,
    ttir.IndexSelectOp: select_golden,
    ttir.IndexOp: index_golden,
    ttir.SliceOp: slice_golden,
    ttir.GatherOp: gather_golden,
    # Neural network operations
    ttir.SoftmaxOp: softmax_golden,
    ttir.MatmulOp: torch.matmul,
    ttir.EmbeddingOp: embedding_golden,
    ttir.Upsample2dOp: upsample2d_golden,
    ttir.BatchNormOp: batch_norm_golden,
    # Type operations
    ttir.TypecastOp: torch.Tensor.type,
    # Tensor creation
    ttir.ZerosOp: zeros_golden,
    ttir.OnesOp: ones_golden,
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
    # Layout operations (identity functions) — accept and ignore extra kwargs like reinterpretLayout
    ttir.ToLayoutOp: (lambda x, **kwargs: x),
    ttir.ViewLayoutOp: (lambda x, **kwargs: x),
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
    ttir.LeakyReluOp: leaky_relu_golden,
    stablehlo.AddOp: torch.add,
}
