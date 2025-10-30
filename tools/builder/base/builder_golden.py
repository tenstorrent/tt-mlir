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

from __future__ import annotations
from typing import Dict, Callable, Any, Optional, Union, List, Tuple, Iterable, Iterator
import itertools
import operator
import torch
import torch.nn.functional
from ttmlir.dialects import ttir, stablehlo, d2m, ttnn
from ttmlir.ir import (
    Attribute,
    ArrayAttr,
    IntegerAttr,
    IntegerType,
    BoolAttr,
    DenseI32ArrayAttr,
    DenseI64ArrayAttr,
)


class BuilderGoldenTensor:
    """
    BuilderGoldenTensor is a utility class for managing golden values for single device or multi device tensors.
    BuilderGoldenTensor represents a list of torch.Tensor objects, each representing a shard of a tensor.
    For single device tensors, it contains a single shard.

    How is this class compatible with torch.* operations?
      * For read-only tensor attributes (like shape, dtype, device, etc.), BuilderGoldenTensor forwards attribute access to the first shard.
      * For torch.* operations, the class implements the `__torch_function__` protocol.
      * When a torch function is called on a BuilderGoldenTensor, the function is applied independently to each shard.
      * The results are collected into a new BuilderGoldenTensor.
    """

    # Names that are safe to forward to the first shard.
    _SAFE_TENSOR_ATTRS = {
        "shape",
        "dtype",
        "device",
        "layout",
        "requires_grad",
        "is_cuda",
        "is_floating_point",
        "is_complex",
        "ndim",
        "dim",
        "size",
        "stride",
        "storage_offset",
        "numel",
        "element_size",
        "is_contiguous",
        "is_pinned",
        "is_quantized",
    }

    # Mutating methods. We will always return a new BuilderGoldenTensor to avoid in-place mutations by design.
    _MUTATING_METHOD_NAMES = [
        "to",
        "transpose",
        "reshape",
        "repeat",
        "permute",
        "flatten",
        "squeeze",
        "unsqueeze",
        "float",
        "clamp",
        "int_repr",
        "detach",
        "requires_grad_",
        "long",
    ]
    _MUTATING_METHODS = {
        name: (
            lambda name: lambda shard, *args, **kwargs: getattr(shard, name)(
                *args, **kwargs
            )
        )(name)
        for name in _MUTATING_METHOD_NAMES
    }

    # ----- Methods -----

    def __init__(self, shard_map: Dict[int, torch.Tensor], mesh_shape: Tuple[int, int]):
        it = iter(shard_map.values())
        first = next(it)

        for t in it:
            if t.shape != first.shape:
                raise ValueError(f"Shape mismatch: {t.shape} != {first.shape}")
            if t.dtype != first.dtype:
                raise ValueError(f"Dtype mismatch: {t.dtype} != {first.dtype}")
            if t.device != first.device:
                raise ValueError(f"Device mismatch: {t.device} != {first.device}")

        self._shard_map = shard_map
        self._mesh_shape = mesh_shape

    # ----- Private static methods -----
    def __getitem__(self, key: int) -> BuilderGoldenTensor:
        out_shards = {k: v.__getitem__(key) for k, v in self._shard_map.items()}
        ref = next(iter(out_shards.values()))
        if not all(isinstance(t, torch.Tensor) for t in out_shards.values()):
            return out_shards
        # Wrap
        return BuilderGoldenTensor(out_shards, self.mesh_shape)

    def _binary_map(self, other, op):
        if isinstance(other, BuilderGoldenTensor):
            keys = sorted(self._shard_map.keys())
            if set(keys) != set(other._shard_map.keys()):
                raise RuntimeError("Shard key mismatch between operands.")
            out_shards = {k: op(self._shard_map[k], other._shard_map[k]) for k in keys}
        else:
            out_shards = {k: op(t, other) for k, t in self._shard_map.items()}
        # Always wrap (even 0-D)
        return BuilderGoldenTensor(out_shards, self._mesh_shape)

    def __lt__(self, other):
        return self._binary_map(other, operator.lt)

    def __le__(self, other):
        return self._binary_map(other, operator.le)

    def __gt__(self, other):
        return self._binary_map(other, operator.gt)

    def __ge__(self, other):
        return self._binary_map(other, operator.ge)

    def __eq__(self, other):
        return self._binary_map(other, operator.eq)

    def __ne__(self, other):
        return self._binary_map(other, operator.ne)

    def __add__(self, other):
        return self._binary_map(other, operator.add)

    def __radd__(self, other):
        return self._binary_map(other, operator.add)

    def __sub__(self, other):
        return self._binary_map(other, operator.sub)

    def __rsub__(self, other):
        return self._binary_map(other, lambda a, b: operator.sub(b, a))

    @staticmethod
    def _walk_tree(*trees) -> Iterable:
        # Yield leaves in a nested structure.
        for tree in trees:
            if isinstance(tree, (list, tuple)):
                for v in tree:
                    yield from BuilderGoldenTensor._walk_tree(v)
            elif isinstance(tree, dict):
                for v in tree.values():
                    yield from BuilderGoldenTensor._walk_tree(v)
            else:
                yield tree

    # ----- Private methods -----

    def __getattr__(self, name: str) -> Any:
        if name in BuilderGoldenTensor._SAFE_TENSOR_ATTRS:
            return getattr(self._shard_map[0], name)
        elif name in BuilderGoldenTensor._MUTATING_METHODS:

            def method(*args, **kwargs):
                func = BuilderGoldenTensor._MUTATING_METHODS[name]
                return self.apply_shardwise(lambda shard: func(shard, *args, **kwargs))

            return method
        raise AttributeError(
            f"'BuilderGoldenTensor' has no attribute '{name}'. "
            "For mutating ops call torch.* directly."
        )

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if not any(issubclass(t, cls) for t in types):
            return NotImplemented

        # Collect all BuilderGoldenTensor inputs.
        st_inputs = [
            a
            for a in BuilderGoldenTensor._walk_tree(args, kwargs)
            if isinstance(a, cls)
        ]
        shard_counts = {len(st.shard_map) for st in st_inputs}
        if len(shard_counts) != 1:
            raise RuntimeError(
                "All BuilderGoldenTensors must have the same shard count."
            )

        # All shard_maps should share the same set of keys.
        shard_keys = [set(st.shard_map.keys()) for st in st_inputs]
        if not all(keys == shard_keys[0] for keys in shard_keys[1:]):
            raise RuntimeError(
                "All BuilderGoldenTensors must have the same shard keys (devices amongst which the BuilderGoldenTensor lives)."
            )

        keys = sorted(shard_keys[0])  # deterministic order

        def _take(obj, k: int):
            if isinstance(obj, cls):
                return obj.shard_map[k]
            if isinstance(obj, (list, tuple)):
                return type(obj)(_take(v, k) for v in obj)
            if isinstance(obj, dict):
                return {kk: _take(v, k) for kk, v in obj.items()}
            return obj

        # Apply func shard-wise.
        out_shards = {k: func(*_take(args, k), **_take(kwargs, k)) for k in keys}

        # Check if output is a tuple of tensors
        if isinstance(next(iter(out_shards.values())), tuple):
            # Wrap each element of the tuple separately
            tuple_len = len(next(iter(out_shards.values())))
            wrapped_tuple = []
            for i in range(tuple_len):
                element_shards = {k: out_shards[k][i] for k in keys}
                wrapped_tuple.append(cls(element_shards, st_inputs[0].mesh_shape))
            return tuple(wrapped_tuple)

        # If all results are Tensors and compatible, wrap back into BuilderGoldenTensor.
        if all(isinstance(o, torch.Tensor) for o in out_shards.values()):
            ref = next(iter(out_shards.values()))
            if all(
                (o.shape, o.dtype, o.device) == (ref.shape, ref.dtype, ref.device)
                for o in out_shards.values()
            ):
                return cls(out_shards, st_inputs[0].mesh_shape)

        return out_shards

    # ----- Public methods -----

    @property
    def shard_map(self) -> Dict[int, torch.Tensor]:
        return self._shard_map

    @property
    def mesh_shape(self) -> Tuple[int, int]:
        return self._mesh_shape

    def zeros_like_builder(self, shape) -> BuilderGoldenTensor:
        shard_map = {}
        for device_id, shard in self.shard_map.items():
            shard_map[device_id] = torch.zeros(shape, dtype=shard.dtype)
        return BuilderGoldenTensor(shard_map, self.mesh_shape)

    def apply_shardwise(
        input_tensor: BuilderGoldenTensor, func: Callable
    ) -> BuilderGoldenTensor:
        shard_map = {}
        for device_id, shard in input_tensor.shard_map.items():
            output_shard = shard.clone()
            shard_map[device_id] = func(output_shard)
        return BuilderGoldenTensor(shard_map, input_tensor.mesh_shape)

    def shard_at(self, device_id: int) -> BuilderGoldenTensor:
        if device_id not in self._shard_map:
            raise KeyError(f"Device ID {device_id} not found in shard map.")
        return self._shard_map[device_id]

    def clone(self) -> BuilderGoldenTensor:
        shard_map = {
            device_id: shard.clone() for device_id, shard in self.shard_map.items()
        }
        return BuilderGoldenTensor(shard_map, self.mesh_shape)

    def contiguous(self) -> BuilderGoldenTensor:
        return BuilderGoldenTensor(
            {k: t.contiguous() for k, t in self._shard_map.items()}, self.mesh_shape
        )

    def group_by_axis(self, axis: int) -> List[Dict[int, torch.Tensor]]:
        rows, cols = self._mesh_shape
        shard_map = self._shard_map

        grouped: List[Dict[int, torch.Tensor]] = []
        if axis == 1:
            for r in range(rows):
                row_group: Dict[int, torch.Tensor] = {}
                for c in range(cols):
                    idx = r * cols + c
                    row_group[idx] = shard_map[idx]
                grouped.append(row_group)
        else:
            for c in range(cols):
                col_group: Dict[int, torch.Tensor] = {}
                for r in range(rows):
                    idx = r * cols + c
                    col_group[idx] = shard_map[idx]
                grouped.append(col_group)

        return grouped


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


def cbrt_golden(x: BuilderGoldenTensor) -> BuilderGoldenTensor:
    """
    Custom golden function for cubic root.

    Parameters
    ----------
    x : BuilderGoldenTensor
        Input tensor

    Returns
    -------
    BuilderGoldenTensor
        BuilderGoldenTensor containing the cubic root of each element in the input tensor
    """
    golden_sign = torch.sign(x)
    golden_cbrt = torch.pow(torch.abs(x), 1 / 3)
    return torch.mul(golden_sign, golden_cbrt)


def conv2d_golden(
    input_tensor: BuilderGoldenTensor,
    weight: BuilderGoldenTensor,
    bias: Optional[BuilderGoldenTensor] = None,
    **kwargs,
) -> BuilderGoldenTensor:
    """
    Custom golden function for conv2d with layout transformation.

    Parameters
    ----------
    input_tensor : BuilderGoldenTensor
        Input tensor for convolution
    weight : BuilderGoldenTensor
        Convolution weight tensor
    bias : BuilderGoldenTensor, optional
        Optional bias tensor (default: None)
    **kwargs : dict
        Keyword arguments containing:
        - stride: Union[int, List[int]] - Stride for convolution (default: 1)
        - padding: Union[int, List[int]] - Padding for convolution (default: 0)
        - dilation: Union[int, List[int]] - Dilation for convolution (default: 1)
        - groups: int - Number of groups for grouped convolution (default: 1)

    Returns
    -------
    BuilderGoldenTensor
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
    copied_input_tensor = input_tensor.clone()
    copied_input_tensor = copied_input_tensor.transpose(-2, -1).transpose(-3, -2)

    if copied_input_tensor.is_quantized:
        if not weight.is_quantized:
            raise ValueError("Quantized input requires quantized weight.")
        # if input tensor and weight tensor zero points are different, error out
        if (copied_input_tensor.q_zero_point() - 128) != weight.q_zero_point():
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
            copied_input_tensor,
            packed_weight,
            copied_input_tensor.q_scale() * weight.q_scale(),
            copied_input_tensor.q_zero_point(),
        ).int_repr()

    else:
        if bias is not None:
            bias = bias.squeeze()

        result = torch.nn.functional.conv2d(
            copied_input_tensor,
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
    input_tensor: BuilderGoldenTensor,
    weight: BuilderGoldenTensor,
    bias: Optional[BuilderGoldenTensor] = None,
    **kwargs,
) -> BuilderGoldenTensor:
    """
    Custom golden function for conv_transpose2d with layout transformation.

    Parameters
    ----------
    input_tensor : BuilderGoldenTensor
        Input tensor for transposed convolution
    weight : BuilderGoldenTensor
        Convolution weight tensor
    bias : Optional[BuilderGoldenTensor]
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
    BuilderGoldenTensor
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
    copied_input_tensor = input_tensor.clone()
    copied_input_tensor = copied_input_tensor.transpose(-2, -1).transpose(-3, -2)
    result = torch.nn.functional.conv_transpose2d(
        copied_input_tensor,
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


def max_pool2d_golden(
    input_tensor: BuilderGoldenTensor, **kwargs
) -> BuilderGoldenTensor:
    """
    Custom golden function for max_pool2d with layout transformation.

    Parameters
    ----------
    input_tensor : BuilderGoldenTensor
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
    BuilderGoldenTensor
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


def avg_pool2d_golden(
    input_tensor: BuilderGoldenTensor, **kwargs
) -> BuilderGoldenTensor:
    """
    Custom golden function for max_pool2d with layout transformation.

    Parameters
    ----------
    input_tensor : BuilderGoldenTensor
        Input tensor for max pooling
    **kwargs : dict
        Keyword arguments containing:
        - kernel_size: Union[int, List[int]] - Size of the pooling kernel
        - stride: Union[int, List[int]] - Stride for pooling operation
        - padding: Union[int, List[int]] - Padding for pooling operation
        - dilation: Union[int, List[int]] - Dilation for pooling operation
        - ceil_mode: bool - Whether to use ceiling mode for pooling
        - count_include_pad: bool - Whether to include padding in the average calculation

    Returns
    -------
    BuilderGoldenTensor
        Result of 2D max pooling with layout transformation
    """
    # Get parameters from ttir_kwargs
    kernel_size = kwargs.get("kernel")
    stride = kwargs.get("stride", kernel_size)  # Default stride = kernel size
    padding = kwargs.get("padding", 0)
    dilation = kwargs.get("dilation", 1)
    ceil_mode = kwargs.get("ceil_mode", False)
    count_include_pad = kwargs.get("count_include_pad", True)

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
    if dilation != [1, 1]:
        raise ValueError("Dilation is not supported for torch.nn.AvgPool2d")
    maxpool_object = torch.nn.AvgPool2d(
        kernel_size, stride, torch_padding, ceil_mode, count_include_pad
    )
    input_tensor = input_tensor.transpose(-2, -1).transpose(-3, -2)
    result = maxpool_object(input_tensor)
    result = result.transpose(-3, -2).transpose(-2, -1)
    return result


def batch_norm_golden(
    input_tensor: BuilderGoldenTensor,
    scale: BuilderGoldenTensor,
    offset: BuilderGoldenTensor,
    mean: BuilderGoldenTensor,
    variance: BuilderGoldenTensor,
    epsilon: float = 1e-5,
    training: bool = False,
    dim: int = 1,
) -> BuilderGoldenTensor:
    """
    Custom golden function for batch normalization with layout transformation.

    Parameters
    ----------
    input_tensor : BuilderGoldenTensor
        Input tensor for batch normalization
    scale : BuilderGoldenTensor
        Scale tensor for batch normalization
    offset : BuilderGoldenTensor
        Offset tensor for batch normalization
    mean : BuilderGoldenTensor
        Mean tensor for batch normalization
    variance : BuilderGoldenTensor
        Variance tensor for batch normalization
    epsilon : float, optional
        Small value to avoid division by zero (default: 1e-5)
    training : bool, optional
        Whether the model is in training mode (default: False)
    dim : int, optional
        Dimension to apply batch normalization over (default: 1)

    Returns
    -------
    BuilderGoldenTensor
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


def rms_norm_golden(
    input: BuilderGoldenTensor,
    weight: Optional[BuilderGoldenTensor] = None,
    bias: Optional[BuilderGoldenTensor] = None,
    normalized_shape: List[int] = None,
    epsilon: float = 1e-5,
) -> BuilderGoldenTensor:
    """
    Custom golden function for RMS normalization operation.
    Parameters
    ----------
    input : BuilderGoldenTensor
        Input tensor to RMS normalization operation
    weight : BuilderGoldenTensor, optional
        Weight tensor for scaling (default: None)
    bias : BuilderGoldenTensor, optional
        Bias tensor for shifting (default: None)
    normalized_shape : List[int], optional
        Shape of the input tensor to normalize (default: None)
    epsilon : float, optional
        Small value to avoid division by zero (default: 1e-5)
    Returns
    -------
    BuilderGoldenTensor
        RMS normalized output tensor
    """
    # Convert to float for computation
    input_float = input.float()

    rms_norm = torch.nn.functional.rms_norm(
        input_float,
        normalized_shape=normalized_shape,
        weight=weight,
        eps=epsilon,
    )

    # Apply bias (shift) if provided
    if bias is not None:
        rms_norm = torch.add(rms_norm, bias.float())

    # Convert back to original dtype
    return rms_norm.to(input.dtype)


def typecast_golden(input_tensor: BuilderGoldenTensor, dtype) -> BuilderGoldenTensor:
    """
    Custom golden function for typecasting.

    Parameters
    ----------
    input_tensor : BuilderGoldenTensor
        Input tensor to typecast
    dtype : torch.dtype
        Target data type for typecasting

    Returns
    -------
    BuilderGoldenTensor
        Typecasted tensor
    """
    return input_tensor.to(dtype)


def argmax_golden(
    input_tensor: BuilderGoldenTensor, dim_arg, keep_dim=False
) -> BuilderGoldenTensor:
    """
    Custom golden function for argmax.

    Parameters
    ----------
    input_tensor : BuilderGoldenTensor
        Input tensor to find argmax of
    dim_arg : List[int]
        List containing dimension to find argmax along
    keep_dim : bool, optional
        Whether to keep the reduced dimension (default: False)

    Returns
    -------
    BuilderGoldenTensor
        Indices of maximum values along specified dimension as int32 tensor
    """
    result = torch.argmax(input_tensor, dim=dim_arg[0], keepdim=keep_dim)
    return result.to(torch.int32)


def linear_golden(
    a: BuilderGoldenTensor,
    b: BuilderGoldenTensor,
    bias=None,
    transpose_a=False,
    transpose_b=False,
) -> BuilderGoldenTensor:
    """
    Custom golden function for linear transformation.

    Parameters
    ----------
    a : BuilderGoldenTensor
        First input tensor
    b : BuilderGoldenTensor
        Second input tensor
    bias : BuilderGoldenTensor, optional
        Optional bias tensor (default: None)
    transpose_a : bool, optional
        Whether to transpose tensor a (default: False)
    transpose_b : bool, optional
        Whether to transpose tensor b (default: False)

    Returns
    -------
    BuilderGoldenTensor
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
    lhs: BuilderGoldenTensor,
    rhs: BuilderGoldenTensor,
    batch_dims_lhs,
    contract_dims_lhs,
    batch_dims_rhs,
    contract_dims_rhs,
) -> BuilderGoldenTensor:
    """
    Custom golden function for dot_general operation.

    Parameters
    ----------
    lhs : BuilderGoldenTensor
        Left-hand side tensor
    rhs : BuilderGoldenTensor
        Right-hand side tensor
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
    BuilderGoldenTensor
        Result of generalized dot product operation
    """
    non_batch_dims_lhs = [d for d in range(lhs.dim()) if d not in batch_dims_lhs]
    non_batch_dims_rhs = [d for d in range(rhs.dim()) if d not in batch_dims_rhs]

    # Compute output shape
    lhs_shape = list(lhs.shape)
    rhs_shape = list(rhs.shape)
    batch_shape = [lhs_shape[d] for d in batch_dims_lhs]
    non_contract_lhs = [d for d in non_batch_dims_lhs if d not in contract_dims_lhs]
    non_contract_rhs = [d for d in non_batch_dims_rhs if d not in contract_dims_rhs]
    out_shape = (
        batch_shape
        + [lhs_shape[d] for d in non_contract_lhs]
        + [rhs_shape[d] for d in non_contract_rhs]
    )

    transposed_lhs = torch.permute(lhs, (batch_dims_lhs + non_batch_dims_lhs))
    transposed_rhs = torch.permute(rhs, (batch_dims_rhs + non_batch_dims_rhs))
    result = lhs.zeros_like_builder(out_shape)

    dim_ranges = []
    for i in range(len(batch_dims_lhs)):
        dim_ranges.append([j for j in range(list(lhs.shape)[i])])

    import itertools

    batch_indices = list(itertools.product(*dim_ranges))
    for index in batch_indices:
        for device_id, shard in result.shard_map.items():
            transposed_lhs_slice = transposed_lhs.shard_at(device_id)[index]
            transposed_rhs_slice = transposed_rhs.shard_at(device_id)[index]
            dot_dims_lhs = [d - len(index) for d in contract_dims_lhs]
            dot_dims_rhs = [d - len(index) for d in contract_dims_rhs]
            out_index = index
            shard[out_index] = torch.tensordot(
                transposed_lhs_slice,
                transposed_rhs_slice,
                dims=(dot_dims_lhs, dot_dims_rhs),
            )
    return result


def quantize_golden(
    input_tensor: BuilderGoldenTensor, scale, zero_point, dtype
) -> BuilderGoldenTensor:
    """
    Custom golden function for quantize operation.

    Parameters
    ----------
    input_tensor : BuilderGoldenTensor
        Input tensor to quantize
    scale : float
        Scale factor for quantization
    zero_point : int
        Zero point for quantization
    dtype : torch.dtype
        Target quantized data type

    Returns
    -------
    BuilderGoldenTensor
        Quantized tensor as integer representation
    """
    return torch.quantize_per_tensor(input_tensor, scale, zero_point, dtype).int_repr()


def requantize_golden(
    input_tensor: BuilderGoldenTensor, scale, zero_point, dtype
) -> BuilderGoldenTensor:
    """
    Custom golden function for requantize operation.

    Parameters
    ----------
    input_tensor : BuilderGoldenTensor
        Input quantized tensor to requantize
    scale : float
        Scale factor for requantization
    zero_point : int
        Zero point for requantization
    dtype : torch.dtype
        Target quantized data type

    Returns
    -------
    BuilderGoldenTensor
        Requantized tensor
    """
    return torch.quantize_per_tensor(
        torch.dequantize(input_tensor), scale, zero_point, dtype
    )


def logical_not_golden(
    input_tensor: BuilderGoldenTensor, **kwargs
) -> BuilderGoldenTensor:
    """
    Golden function for logical_not operation.

    Elementwise logical NOT.

    Parameters
    ----------
    input_tensor : BuilderGoldenTensor
        Input tensor to invert logically.
    **kwargs : dict
        Keyword arguments (unused for this operation).

    Returns
    -------
    BuilderGoldenTensor
        Tensor with logical NOT of input_tensor, cast back to input dtype.
    """
    # Compute bool result then cast to match input dtype
    result_bool = torch.logical_not(input_tensor)
    return result_bool.to(input_tensor.dtype)


def equal_golden(
    input_tensor: BuilderGoldenTensor, other_tensor: BuilderGoldenTensor, **kwargs
) -> BuilderGoldenTensor:
    """
    Golden function for equal (eq) operation.

    Elementwise equality comparison.

    Parameters
    ----------
    input_tensor : BuilderGoldenTensor
        Left-hand side tensor.
    other_tensor : BuilderGoldenTensor
        Right-hand side tensor.

    Returns
    -------
    BuilderGoldenTensor
        Tensor with the same dtype as input_tensor containing the equality results.
    """
    result_bool = torch.eq(input_tensor, other_tensor)
    return result_bool.to(input_tensor.dtype)


def not_equal_golden(
    input_tensor: BuilderGoldenTensor, other_tensor: BuilderGoldenTensor, **kwargs
) -> BuilderGoldenTensor:
    """
    Golden function for not_equal (ne) operation.

    Elementwise inequality comparison.

    Parameters
    ----------
    input_tensor : BuilderGoldenTensor
        Left-hand side tensor.
    other_tensor : BuilderGoldenTensor
        Right-hand side tensor.

    Returns
    -------
    BuilderGoldenTensor
        Tensor with the same dtype as input_tensor containing the inequality results.
    """
    result_bool = torch.ne(input_tensor, other_tensor)
    return result_bool.to(input_tensor.dtype)


def greater_equal_golden(
    input_tensor: BuilderGoldenTensor, other_tensor: BuilderGoldenTensor, **kwargs
) -> BuilderGoldenTensor:
    """
    Golden function for greater_equal (ge) operation.

    Elementwise greater-than-or-equal comparison.

    Parameters
    ----------
    input_tensor : BuilderGoldenTensor
        Left-hand side tensor.
    other_tensor : BuilderGoldenTensor
        Right-hand side tensor.

    Returns
    -------
    BuilderGoldenTensor
        Tensor with the same dtype as input_tensor containing the comparison results.
    """
    result_bool = torch.ge(input_tensor, other_tensor)
    return result_bool.to(input_tensor.dtype)


def greater_than_golden(
    input_tensor: BuilderGoldenTensor, other_tensor: BuilderGoldenTensor, **kwargs
) -> BuilderGoldenTensor:
    """
    Golden function for greater_than (gt) operation.

    Elementwise greater-than comparison.

    Parameters
    ----------
    input_tensor : BuilderGoldenTensor
        Left-hand side tensor.
    other_tensor : BuilderGoldenTensor
        Right-hand side tensor.

    Returns
    -------
    BuilderGoldenTensor
        Tensor with the same dtype as input_tensor containing the comparison results.
    """
    result_bool = torch.gt(input_tensor, other_tensor)
    return result_bool.to(input_tensor.dtype)


def less_equal_golden(
    input_tensor: BuilderGoldenTensor, other_tensor: BuilderGoldenTensor, **kwargs
) -> BuilderGoldenTensor:
    """
    Golden function for less_equal (le) operation.

    Elementwise less-than-or-equal comparison.

    Parameters
    ----------
    input_tensor : BuilderGoldenTensor
        Left-hand side tensor.
    other_tensor : BuilderGoldenTensor
        Right-hand side tensor.

    Returns
    -------
    BuilderGoldenTensor
        Tensor with the same dtype as input_tensor containing the comparison results.
    """
    result_bool = torch.le(input_tensor, other_tensor)
    return result_bool.to(input_tensor.dtype)


def less_than_golden(
    input_tensor: BuilderGoldenTensor, other_tensor: BuilderGoldenTensor, **kwargs
) -> BuilderGoldenTensor:
    """
    Golden function for less_than (lt) operation.

    Elementwise less-than comparison.

    Parameters
    ----------
    input_tensor : BuilderGoldenTensor
        Left-hand side tensor.
    other_tensor : BuilderGoldenTensor
        Right-hand side tensor.

    Returns
    -------
    BuilderGoldenTensor
        Tensor with the same dtype as input_tensor containing the comparison results.
    """
    result_bool = torch.lt(input_tensor, other_tensor)
    return result_bool.to(input_tensor.dtype)


def logical_and_golden(
    input_tensor: BuilderGoldenTensor, other_tensor: BuilderGoldenTensor, **kwargs
) -> BuilderGoldenTensor:
    """
    Golden function for logical_and operation.

    Elementwise logical AND.

    Parameters
    ----------
    input_tensor : BuilderGoldenTensor
        Left-hand side tensor.
    other_tensor : BuilderGoldenTensor
        Right-hand side tensor.

    Returns
    -------
    BuilderGoldenTensor
        Tensor with the same dtype as input_tensor containing the logical AND results.
    """
    result_bool = torch.logical_and(input_tensor, other_tensor)
    return result_bool.to(input_tensor.dtype)


def logical_or_golden(
    input_tensor: BuilderGoldenTensor, other_tensor: BuilderGoldenTensor, **kwargs
) -> BuilderGoldenTensor:
    """
    Golden function for logical_or operation.

    Elementwise logical OR.

    Parameters
    ----------
    input_tensor : BuilderGoldenTensor
        Left-hand side tensor.
    other_tensor : BuilderGoldenTensor
        Right-hand side tensor.

    Returns
    -------
    BuilderGoldenTensor
        Tensor with the same dtype as input_tensor containing the logical OR results.
    """
    result_bool = torch.logical_or(input_tensor, other_tensor)
    return result_bool.to(input_tensor.dtype)


def logical_xor_golden(
    input_tensor: BuilderGoldenTensor, other_tensor: BuilderGoldenTensor, **kwargs
) -> BuilderGoldenTensor:
    """
    Golden function for logical_xor operation.

    Elementwise logical XOR.

    Parameters
    ----------
    input_tensor : BuilderGoldenTensor
        Left-hand side tensor.
    other_tensor : BuilderGoldenTensor
        Right-hand side tensor.

    Returns
    -------
    BuilderGoldenTensor
        Tensor with the same dtype as input_tensor containing the logical XOR results.
    """
    result_bool = torch.logical_xor(input_tensor, other_tensor)
    return result_bool.to(input_tensor.dtype)


def logical_left_shift_golden(
    input_tensor: BuilderGoldenTensor, shift_tensor: BuilderGoldenTensor, **kwargs
) -> BuilderGoldenTensor:
    """
    Golden function for logical left shift operation.

    Parameters
    ----------
    input_tensor : BuilderGoldenTensor
        Input tensor to be shifted.
    shift_tensor : BuilderGoldenTensor
        Tensor containing the number of bits to shift.

    Returns
    -------
    BuilderGoldenTensor
        Tensor with the same dtype as input_tensor after logical left shift.
    """
    # Perform logical left shift
    # Convert both inputs to int64 to handle both signed and unsigned types
    input_int64 = input_tensor.to(torch.int64)
    shift_int64 = shift_tensor.to(torch.int64)

    # Mask input to 32-bit unsigned range (for signed types this converts to unsigned interpretation)
    input_unsigned = torch.bitwise_and(input_int64, 0xFFFFFFFF)

    # Perform shift in int64 space
    result = torch.bitwise_left_shift(input_unsigned, shift_int64)

    # Mask result to keep in valid 32-bit range
    result = torch.bitwise_and(result, 0xFFFFFFFF)

    # Convert back to original dtype
    return result.to(input_tensor.dtype)


def logical_right_shift_golden(
    input_tensor: BuilderGoldenTensor, shift_tensor: BuilderGoldenTensor, **kwargs
) -> BuilderGoldenTensor:
    """
    Golden function for logical right shift operation.

    Parameters
    ----------
    input_tensor : BuilderGoldenTensor
        Input tensor to be shifted.
    shift_tensor : BuilderGoldenTensor
        Tensor containing the number of bits to shift.

    Returns
    -------
    BuilderGoldenTensor
        Tensor with the same dtype as input_tensor after logical right shift.
    """
    # Perform logical (unsigned) right shift
    # Convert both inputs to int64 to handle both signed and unsigned types
    input_int64 = input_tensor.to(torch.int64)
    shift_int64 = shift_tensor.to(torch.int64)

    # Mask input to 32-bit unsigned range (for signed types this converts to unsigned interpretation)
    input_unsigned = torch.bitwise_and(input_int64, 0xFFFFFFFF)

    # Perform shift in int64 space
    result = torch.bitwise_right_shift(input_unsigned, shift_int64)

    # Mask result to keep in valid range
    result = torch.bitwise_and(result, 0xFFFFFFFF)

    # Convert back to original dtype
    return result.to(input_tensor.dtype)


def max_golden(
    input_tensor: BuilderGoldenTensor, dim_arg=None, keep_dim=True
) -> BuilderGoldenTensor:
    """
    Custom golden function for max operation with conditional logic.

    Parameters
    ----------
    input_tensor : BuilderGoldenTensor
        Input tensor to find maximum of
    dim_arg : int, optional
        Dimension to find maximum along (default: None for all dimensions)
    keep_dim : bool, optional
        Whether to keep the reduced dimension (default: True)

    Returns
    -------
    BuilderGoldenTensor
        Maximum values along specified dimension or global maximum
    """
    if dim_arg is not None:
        values, indices = torch.max(input_tensor, dim=dim_arg, keepdim=keep_dim)
        return values
    else:
        # For all dimensions reduction, reshape to match expected output
        result = torch.max(input_tensor)
        output_shape = [1] * input_tensor.dim()
        return result.reshape(*output_shape)


def min_golden(input_tensor: BuilderGoldenTensor, **kwargs) -> BuilderGoldenTensor:
    """
    Golden function for min operation.

    Parameters
    ----------
    input_tensor : BuilderGoldenTensor
        Input tensor
    **kwargs : dict
        Keyword arguments containing:
        - dim_arg: int, optional - Dimension to reduce over (default: None, reduces over all dimensions)
        - keep_dim: bool, optional - If True, retains reduced dimensions with length 1 (default: True)

    Returns
    -------
    BuilderGoldenTensor
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


def prod_golden(
    input_tensor: BuilderGoldenTensor, dim_arg, keep_dim=False
) -> BuilderGoldenTensor:
    """
    Custom golden function for prod operation with conditional logic.

    Parameters
    ----------
    input_tensor : BuilderGoldenTensor
        Input tensor to compute product of
    dim_arg : List[int]
        List of dimensions to compute product along
    keep_dim : bool, optional
        Whether to keep the reduced dimension (default: False)

    Returns
    -------
    BuilderGoldenTensor
        Product of tensor elements along specified dimensions
    """
    if len(dim_arg) == 1:
        return torch.prod(input_tensor, dim=dim_arg[0], keepdim=keep_dim)
    else:
        # Multiple dimensions - reduce to scalar
        output_tensor = input_tensor.clone()

        for device_id, shard in output_tensor.shard_map.items():
            shard = torch.tensor([torch.prod(input_tensor.shard_at(device_id)).item()])

        return output_tensor


def embedding_golden(
    indices_tensor: BuilderGoldenTensor, weight_tensor: BuilderGoldenTensor
) -> BuilderGoldenTensor:
    """
    Custom golden function for embedding operation.

    Parameters
    ----------
    indices_tensor : BuilderGoldenTensor
        Tensor containing indices to look up
    weight_tensor : BuilderGoldenTensor
        Weight tensor containing embedding vectors

    Returns
    -------
    BuilderGoldenTensor
        Embedded vectors corresponding to input indices
    """
    embedding = torch.nn.Embedding.from_pretrained(weight_tensor)
    golden_typecast = indices_tensor.to(torch.int32)
    golden_input = torch.clamp(golden_typecast, 0, (weight_tensor.size()[0] - 1))
    return embedding(golden_input)


def pad_golden(
    input_tensor: BuilderGoldenTensor, padding, value
) -> BuilderGoldenTensor:
    """
    Custom golden function for pad operation with dimension reformatting.

    Parameters
    ----------
    input_tensor : BuilderGoldenTensor
        Input tensor to pad
    padding : List[int]
        Padding specification
    value : Union[int, float]
        Value to use for padding

    Returns
    -------
    BuilderGoldenTensor
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


def select_golden(
    input_tensor: BuilderGoldenTensor, dim, begin, length, stride
) -> BuilderGoldenTensor:
    """
    Custom golden function for select operation.

    Parameters
    ----------
    input_tensor : BuilderGoldenTensor
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
    BuilderGoldenTensor
        Selected tensor slice
    """
    end = begin + length - 1
    index = torch.tensor([begin, end])
    return torch.index_select(input_tensor, dim=dim, index=index)


def index_golden(
    input_tensor: BuilderGoldenTensor, dim, begin, end, step
) -> BuilderGoldenTensor:
    """
    Custom golden function for index operation.

    Parameters
    ----------
    input_tensor : BuilderGoldenTensor
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
    BuilderGoldenTensor
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
    input_tensor: BuilderGoldenTensor,
    start_indices_tensor: BuilderGoldenTensor,
    **kwargs,
) -> BuilderGoldenTensor:
    """
    Golden function for gather operation with TTIR parameter names.

    Parameters
    ----------
    input_tensor : BuilderGoldenTensor
        Input tensor to gather from
    start_indices_tensor : BuilderGoldenTensor
        Tensor containing starting indices
    **kwargs : dict
        Keyword arguments including gather attributes as MLIR attributes

    Returns
    -------
    BuilderGoldenTensor
        Gathered tensor
    """

    # helpers
    def _isbuildergoldentensor(x):
        return isinstance(x, BuilderGoldenTensor)

    def _first_shard(x):
        return x.shard_at(0) if _isbuildergoldentensor(x) else x

    def _assert_replicated(t):
        ref = t.shard_at(0)
        for device_id, shard in t.shard_map.items():
            if not torch.equal(shard, ref):
                raise ValueError("gather golden expects replicated tensors")

    # ----- unpack attrs -----
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

    x = input_tensor
    idx = start_indices_tensor
    device = x.device if hasattr(x, "device") else None

    # validate/comute using shard-0 (torch), then slice the wrapper
    if _isbuildergoldentensor(idx):
        _assert_replicated(idx)
    x0 = _first_shard(x)
    idx0 = _first_shard(idx)

    # ---- validate attrs ----
    rank = x0.dim()
    assert len(slice_sizes) == rank, "slice_sizes must match operand dimensions"
    assert set(collapsed_slice_dims) == set(
        start_index_map
    ), "gathe golden assumes collapsed_slice_dims == start_index_map"
    assert (
        len(operand_batching_dims) == 0 and len(start_indices_batching_dims) == 0
    ), "Batching dims not supported in this golden"
    for d in collapsed_slice_dims:
        assert slice_sizes[d] == 1, "collapsed dims must have slice size 1"

    if idx0.dim() == 0:
        idx0 = idx0.unsqueeze(0)
    if len(start_index_map) == 1 and index_vector_dim == idx0.ndim:
        pass
    else:
        # Expect the conventional "last dim holds the vector"
        assert (
            index_vector_dim == idx0.ndim - 1
        ), "This golden expects index_vector_dim == last dimension for multi-d indices"

    # Determine batch shape and flatten indices to [B, K]
    if idx0.ndim == 1:  # simple path, K == 1
        batch_shape = idx0.shape  # [N]
        K = 1
        idx_flat0 = idx0.reshape(-1, 1).long()
    else:
        K = idx0.shape[-1]
        assert K == len(
            start_index_map
        ), "index vector length must match start_index_map"
        batch_shape = idx0.shape[:-1]
        idx_flat0 = idx0.reshape(-1, K).long()

    # Bounds check (might help avoid segfaults)
    for d in range(rank):
        if d not in start_index_map:
            assert slice_sizes[d] <= x0.size(d), "slice size too large for operand"
    for k, d in enumerate(start_index_map):
        valid_max = x0.size(d) - slice_sizes[d]
        if torch.any(idx_flat0[:, k] < 0) or torch.any(idx_flat0[:, k] > valid_max):
            print(d)
            raise IndexError(
                "gather start indices out of bounds for operand dim {}".format(d)
            )

    # Build the natural slice_shape (operand order, skipping collapsed dims)
    slice_dims_natural = [d for d in range(rank) if d not in collapsed_slice_dims]
    natural_slice_shape = [slice_sizes[d] for d in slice_dims_natural]

    # Number of non-collapsed dims must match offset_dims count
    assert len(slice_dims_natural) == len(
        offset_dims
    ), "offset_dims must have one entry per non-collapsed slice dim"

    # For each batch vector of indices, slice x accordingly
    B = int(torch.tensor(batch_shape).prod()) if len(batch_shape) > 0 else 1
    slices = []
    for b in range(B):
        starts = [0] * rank
        ends = [0] * rank
        # Fill starts/ends from index vector for mapped dims
        for k, d in enumerate(start_index_map):
            starts[d] = int(idx_flat0[b, k].item())
            ends[d] = starts[d] + slice_sizes[d]
        # For the other dims, start at 0 (or clamp) and take slice_sizes[d]
        for d in range(rank):
            if d not in start_index_map:
                starts[d] = 0
                ends[d] = slice_sizes[d]

        # Build the per-dim slice
        slicer = tuple(slice(starts[d], ends[d]) for d in range(rank))
        sub = x[slicer]  # shape equals slice_sizes in operand order

        # Remove collapsed dims (size-1) to get natural slice shape
        if len(collapsed_slice_dims) > 0:
            sub = sub.squeeze(dim=tuple(sorted(collapsed_slice_dims)))

        slices.append(sub)

    # Stack over batch
    if len(slices) == 1 and batch_shape == ():
        gathered = slices[0]
    else:
        gathered = torch.stack(slices, dim=0).reshape(
            *batch_shape, *natural_slice_shape
        )

    # position the slice dims inside the result according to offset_dims.
    # Current order: [B0, B1, ..., Slice0, Slice1, ...]
    batch_rank = len(batch_shape)
    slice_rank = len(natural_slice_shape)
    result_rank = batch_rank + slice_rank

    remaining_positions = [p for p in range(result_rank) if p not in offset_dims]
    assert (
        len(remaining_positions) == batch_rank
    ), "offset_dims inconsistent with batch rank"

    desired_index_for_current = [None] * result_rank
    # map batch dims
    for b_i in range(batch_rank):
        desired_index_for_current[b_i] = remaining_positions[b_i]
    # map slice dims
    for s_i in range(slice_rank):
        desired_index_for_current[batch_rank + s_i] = offset_dims[s_i]

    # Permute if needed
    if desired_index_for_current != list(range(result_rank)):
        gathered = gathered.permute(*desired_index_for_current)

    return gathered.to(device=device)


def tilize_golden(
    input_tensor: BuilderGoldenTensor, tilize=True, **kwargs
) -> BuilderGoldenTensor:
    """
    Custom golden function for tilize operation.

    Parameters
    ----------
    input_tensor : BuilderGoldenTensor
        Input tensor to tilize
    tilize : bool, optional
        Tilize parameter (ignored, for compatibility) (default: True)

    Returns
    -------
    BuilderGoldenTensor
        Tilized tensor with proper tile layout transformation
    """
    shape = input_tensor.shape
    TILE_SIZE = 32
    FACE_SIZE = 16
    Y_TILES = shape[0] // TILE_SIZE
    X_TILES = shape[1] // TILE_SIZE
    FACES_PER_TILE = TILE_SIZE // FACE_SIZE

    tilized = input_tensor.zeros_like_builder((input_tensor.numel(),))

    idx = 0
    for tile_y in range(Y_TILES):
        for tile_x in range(X_TILES):
            for face_y in range(FACES_PER_TILE):
                for face_x in range(FACES_PER_TILE):
                    for datum_y in range(FACE_SIZE):
                        for datum_x in range(FACE_SIZE):
                            for device_id, shard in tilized.shard_map.items():
                                shard[idx] = input_tensor.shard_at(device_id)[
                                    datum_y + tile_y * TILE_SIZE + face_y * FACE_SIZE,
                                    datum_x + tile_x * TILE_SIZE + face_x * FACE_SIZE,
                                ]
                            idx += 1

    tilized = tilized.reshape(shape)
    return tilized


def untilize_golden(
    input_tensor: BuilderGoldenTensor, tilize=False, **kwargs
) -> BuilderGoldenTensor:
    """
    Custom golden function for untilize operation.

    Parameters
    ----------
    input_tensor : BuilderGoldenTensor
        Input tensor to untilize
    tilize : bool, optional
        Tilize parameter (ignored, for compatibility) (default: False)

    Returns
    -------
    BuilderGoldenTensor
        Untilized tensor with proper layout transformation
    """
    shape = input_tensor.shape
    TILE_SIZE = 32
    FACE_SIZE = 16
    Y_TILES = shape[0] // TILE_SIZE
    X_TILES = shape[1] // TILE_SIZE
    FACES_PER_TILE = TILE_SIZE // FACE_SIZE

    untilized = input_tensor.zeros_like_builder(input_tensor.shape)
    flattened = input_tensor.clone()
    flattened = flattened.flatten()

    idx = 0
    for tile_y in range(Y_TILES):
        for tile_x in range(X_TILES):
            for face_y in range(FACES_PER_TILE):
                for face_x in range(FACES_PER_TILE):
                    for datum_y in range(FACE_SIZE):
                        for datum_x in range(FACE_SIZE):
                            for device_id, shard in untilized.shard_map.items():
                                # Calculate the original position
                                orig_y = (
                                    datum_y + tile_y * TILE_SIZE + face_y * FACE_SIZE
                                )
                                orig_x = (
                                    datum_x + tile_x * TILE_SIZE + face_x * FACE_SIZE
                                )

                                # Place the value from the tilized tensor back to its original position
                                shard[orig_y, orig_x] = flattened.shard_at(device_id)[
                                    idx
                                ]
                            idx += 1

    return untilized


def upsample2d_golden(
    in0: BuilderGoldenTensor, in1: BuilderGoldenTensor, scale_factor, mode="nearest"
) -> BuilderGoldenTensor:
    """
    Custom golden function for upsample2d operation.

    Parameters
    ----------
    in0 : BuilderGoldenTensor
        Input tensor to upsample
    in1 : BuilderGoldenTensor
        Output tensor specification
    scale_factor : Union[int, List[int]]
        Scaling factor for upsampling
    mode : str, optional
        Upsampling mode (default: "nearest")

    Returns
    -------
    BuilderGoldenTensor
        Upsampled 2D tensor
    """
    transposed_golden = torch.transpose(in0, 1, 3)
    golden_output_shape = in1.shape[1:-1]
    output = torch.nn.functional.interpolate(
        transposed_golden, size=golden_output_shape, mode=mode
    )
    return torch.transpose(output, 1, 3)


def fill_cache_golden(
    cache_tensor: BuilderGoldenTensor, input_tensor: BuilderGoldenTensor, **kwargs
) -> BuilderGoldenTensor:
    """
    Custom golden function for fill_cache operation.

    Parameters
    ----------
    cache_tensor : BuilderGoldenTensor
        Cache tensor to fill
    input_tensor : BuilderGoldenTensor
        Input tensor data
    **kwargs : dict
        Additional keyword arguments (batch_offset is ignored)

    Returns
    -------
    BuilderGoldenTensor
        Filled cache tensor
    """
    result = cache_tensor.clone()

    for device_id, shard in result.shard_map.items():
        shard[:, :, : input_tensor.shape[2], :] = input_tensor.shard_at(device_id)
    return result


def update_cache_golden(
    cache_tensor: BuilderGoldenTensor,
    update_tensor: BuilderGoldenTensor,
    indices_tensor,
    **kwargs,
) -> BuilderGoldenTensor:
    """
    Custom golden function for update_cache operation.

    Parameters
    ----------
    cache_tensor : BuilderGoldenTensor
        Cache tensor to update
    update_tensor : BuilderGoldenTensor
        Tensor containing update data
    indices_tensor : BuilderGoldenTensor
        Tensor containing update indices
    **kwargs : dict
        Additional keyword arguments (batch_offset is ignored)

    Returns
    -------
    BuilderGoldenTensor
        Updated cache tensor
    """
    result = cache_tensor.clone()

    for device_id, shard in result.shard_map.items():
        shard[:, :, : update_tensor.shape[2], :] = update_tensor.shard_at(device_id)
    return result


def get_dimension_size_golden(
    input_tensor: BuilderGoldenTensor, **kwargs
) -> BuilderGoldenTensor:
    """
    Golden function for get_dimension_size operation.

    Parameters
    ----------
    input_tensor : BuilderGoldenTensor
        Input tensor to get dimension size from
    **kwargs : dict
        Keyword arguments including 'dimension'

    Returns
    -------
    BuilderGoldenTensor
        Tensor containing the size of the specified dimension as int32
    """
    dimension = kwargs.get("dimension", 0)
    output_tensor = input_tensor.clone()

    for device_id, shard in output_tensor.shard_map.items():
        shard = torch.tensor(
            [input_tensor.shard_at(device_id).size(dimension)], dtype=torch.int32
        )

    return output_tensor


def sum_golden(input_tensor: BuilderGoldenTensor, **kwargs) -> BuilderGoldenTensor:
    """
    Golden function for sum operation with TTIR parameter names.

    Parameters
    ----------
    input_tensor : BuilderGoldenTensor
        Input tensor to sum
    **kwargs : dict
        Keyword arguments containing:
        - dim_arg: List[int] - Dimensions to reduce over (default: [0])
        - keep_dim: bool - If True, retains reduced dimensions with length 1 (default: True)

    Returns
    -------
    BuilderGoldenTensor
        Summed tensor
    """
    # Get parameters from ttir_kwargs
    dim_arg = kwargs.get("dim_arg", [0])
    keep_dim = kwargs.get("keep_dim", True)
    # Convert to torch.sum format
    return torch.sum(input_tensor, dim=dim_arg, keepdim=keep_dim)


def mean_golden(input_tensor: BuilderGoldenTensor, **kwargs) -> BuilderGoldenTensor:
    """
    Golden function for mean operation with TTIR parameter names.

    Parameters
    ----------
    input_tensor : BuilderGoldenTensor
        Input tensor to compute mean of
    **kwargs : dict
        Keyword arguments including 'dim_arg' and 'keep_dim'

    Returns
    -------
    BuilderGoldenTensor
        Mean tensor
    """
    dim_arg = kwargs.get("dim_arg", [0])
    keep_dim = kwargs.get("keep_dim", True)
    return torch.mean(input_tensor, dim=dim_arg, keepdim=keep_dim)


def reduce_and_golden(
    input_tensor: BuilderGoldenTensor, **kwargs
) -> BuilderGoldenTensor:
    """
    Golden function for reduce_and operation with TTIR parameter names.

    Parameters
    ----------
    input_tensor : BuilderGoldenTensor
        Input tensor to reduce
    **kwargs : dict
        Keyword arguments including 'dim_arg' and 'keep_dim'

    Returns
    -------
    BuilderGoldenTensor
        Reduced tensor
    """
    dim_arg = kwargs.get("dim_arg", [0])
    keep_dim = kwargs.get("keep_dim", True)
    return torch.all(input_tensor, dim=tuple(dim_arg), keepdim=keep_dim)


def reduce_or_golden(
    input_tensor: BuilderGoldenTensor, **kwargs
) -> BuilderGoldenTensor:
    """
    Golden function for reduce_or operation with TTIR parameter names.

    Parameters
    ----------
    input_tensor : BuilderGoldenTensor
        Input tensor to reduce
    **kwargs : dict
        Keyword arguments including 'dim_arg' and 'keep_dim'

    Returns
    -------
    BuilderGoldenTensor
        Reduced tensor
    """
    dim_arg = kwargs.get("dim_arg", [0])
    keep_dim = kwargs.get("keep_dim", True)
    return torch.any(input_tensor, dim=tuple(dim_arg), keepdim=keep_dim)


def transpose_golden(
    input_tensor: BuilderGoldenTensor, **kwargs
) -> BuilderGoldenTensor:
    """
    Golden function for transpose operation with TTIR parameter names.

    Parameters
    ----------
    input_tensor : BuilderGoldenTensor
        Input tensor
    **kwargs : dict
        Keyword arguments including 'dim0' and 'dim1'

    Returns
    -------
    BuilderGoldenTensor
        Transposed tensor
    """
    dim0 = kwargs.get("dim0", 0)
    dim1 = kwargs.get("dim1", 1)
    return torch.transpose(input_tensor, dim0, dim1)


def sort_golden(input_tensor: BuilderGoldenTensor, **kwargs) -> BuilderGoldenTensor:
    """
    Golden function for sort operation with TTIR parameter names.

    Parameters
    ----------
    input_tensor : BuilderGoldenTensor
        Input tensor
    **kwargs : dict
        Keyword arguments including 'dim', 'descending', and 'stable'

    Returns
    -------
    BuilderGoldenTensor
        Sorted tensor (values only, indices are discarded)
    """
    dim = kwargs.get("dim", -1)
    descending = kwargs.get("descending", False)
    stable = kwargs.get("stable", False)
    values, indices = torch.sort(
        input_tensor, dim=dim, descending=descending, stable=stable
    )
    return values


def concat_golden(input_tensors: BuilderGoldenTensor, **kwargs) -> BuilderGoldenTensor:
    """
    Golden function for concat operation with TTIR parameter names.

    Parameters
    ----------
    input_tensors : BuilderGoldenTensor
        Input tensors (will be unpacked from tuple)
    **kwargs : dict
        Keyword arguments including 'dim'

    Returns
    -------
    BuilderGoldenTensor
        Concatenated tensor
    """
    dim = kwargs.get("dim", 0)
    if isinstance(input_tensors, tuple):
        return torch.concat(input_tensors, dim=dim)
    else:
        return torch.concat([input_tensors], dim=dim)


# Investigate how repeat works in torch
def repeat_golden(input_tensor: BuilderGoldenTensor, **kwargs) -> BuilderGoldenTensor:
    """
    Golden function for repeat operation with TTIR parameter names.

    Parameters
    ----------
    input_tensor : BuilderGoldenTensor
        Input tensor
    **kwargs : dict
        Keyword arguments including 'repeat_dimensions'

    Returns
    -------
    BuilderGoldenTensor
        Repeated tensor
    """
    repeat_dimensions = kwargs.get("repeat_dimensions", [1])
    return input_tensor.repeat(repeats=repeat_dimensions)


def reshape_golden(input_tensor: BuilderGoldenTensor, **kwargs) -> BuilderGoldenTensor:
    """
    Golden function for reshape operation with TTIR parameter names.

    Parameters
    ----------
    input_tensor : BuilderGoldenTensor
        Input tensor
    **kwargs : dict
        Keyword arguments including 'shape'

    Returns
    -------
    BuilderGoldenTensor
        Reshaped tensor
    """
    shape = kwargs.get("shape", input_tensor.shape)
    return torch.reshape(input_tensor, shape)


def squeeze_golden(input_tensor: BuilderGoldenTensor, **kwargs) -> BuilderGoldenTensor:
    """
    Golden function for squeeze operation with TTIR parameter names.

    Parameters
    ----------
    input_tensor : BuilderGoldenTensor
        Input tensor
    **kwargs : dict
        Keyword arguments including 'dim'

    Returns
    -------
    BuilderGoldenTensor
        Squeezed tensor
    """
    dim = kwargs.get("dim", None)
    return torch.squeeze(input_tensor, dim=dim)


def unsqueeze_golden(
    input_tensor: BuilderGoldenTensor, **kwargs
) -> BuilderGoldenTensor:
    """
    Golden function for unsqueeze operation with TTIR parameter names.

    Parameters
    ----------
    input_tensor : BuilderGoldenTensor
        Input tensor
    **kwargs : dict
        Keyword arguments including 'dim'

    Returns
    -------
    BuilderGoldenTensor
        Unsqueezed tensor
    """
    dim = kwargs.get("dim", 0)
    return torch.unsqueeze(input_tensor, dim=dim)


def clamp_scalar_golden(
    input_tensor: BuilderGoldenTensor, **kwargs
) -> BuilderGoldenTensor:
    """
    Golden function for clamp_scalar operation with TTIR parameter names.

    Parameters
    ----------
    input_tensor : BuilderGoldenTensor
        Input tensor
    **kwargs : dict
        Keyword arguments including 'min' and 'max'

    Returns
    -------
    BuilderGoldenTensor
        Clamped tensor
    """
    min_val = kwargs.get("min", None)
    max_val = kwargs.get("max", None)
    return torch.clamp(input_tensor, min=min_val, max=max_val)


def clamp_tensor_golden(
    input_tensor: BuilderGoldenTensor,
    min_tensor: BuilderGoldenTensor,
    max_tensor: BuilderGoldenTensor,
    **kwargs,
) -> BuilderGoldenTensor:
    """
    Golden function for clamp_tensor operation with TTIR parameter names.

    Parameters
    ----------
    input_tensor : BuilderGoldenTensor
        Input tensor
    min_tensor : BuilderGoldenTensor
        Tensor specifying minimum values
    max_tensor : BuilderGoldenTensor
        Tensor specifying maximum values
    **kwargs : dict
        Additional keyword arguments

    Returns
    -------
    BuilderGoldenTensor
        Clamped tensor
    """
    return torch.min(torch.max(input_tensor, min_tensor), max_tensor)


def permute_golden(input_tensor: BuilderGoldenTensor, **kwargs) -> BuilderGoldenTensor:
    """
    Golden function for permute operation with TTIR parameter names.

    Parameters
    ----------
    input_tensor : BuilderGoldenTensor
        Input tensor
    **kwargs : dict
        Keyword arguments including 'permutation' as MLIR attribute

    Returns
    -------
    BuilderGoldenTensor
        Permuted tensor
    """

    permutation = kwargs.get("permutation", None)
    if permutation is None:
        return input_tensor

    permutation = unpack_mlir_attr(permutation)
    return torch.permute(input_tensor, tuple(permutation))


def leaky_relu_golden(
    input_tensor: BuilderGoldenTensor, **kwargs
) -> BuilderGoldenTensor:
    """
    Golden function for leaky_relu operation with TTIR parameter names.

    Parameters
    ----------
    input_tensor : BuilderGoldenTensor
        Input tensor
    **kwargs : dict
        Keyword arguments including 'parameter'

    Returns
    -------
    BuilderGoldenTensor
        Leaky ReLU output
    """
    parameter = kwargs.get("parameter", 0.01)
    return torch.nn.functional.leaky_relu(input_tensor, negative_slope=parameter)


def silu_golden(input_tensor: BuilderGoldenTensor, **kwargs) -> BuilderGoldenTensor:
    """
    Golden function for silu operation with TTIR parameter names.

    Parameters
    ----------
    input_tensor : BuilderGoldenTensor
        Input tensor
    **kwargs : dict
        Additional keyword arguments

    Returns
    -------
    BuilderGoldenTensor
        SiLU output
    """
    return torch.nn.functional.silu(input_tensor)


def softmax_golden(input_tensor: BuilderGoldenTensor, **kwargs) -> BuilderGoldenTensor:
    """
    Golden function for softmax operation with TTIR parameter names.

    Parameters
    ----------
    input_tensor : BuilderGoldenTensor
        Input tensor
    **kwargs : dict
        Keyword arguments including 'dimension'

    Returns
    -------
    BuilderGoldenTensor
        Softmax output
    """
    dimension = kwargs.get("dim", 1)
    return torch.nn.functional.softmax(input_tensor, dim=dimension)


def index_golden(input_tensor: BuilderGoldenTensor, **kwargs) -> BuilderGoldenTensor:
    """
    Golden function for index operation with TTIR parameter names.

    Parameters
    ----------
    input_tensor : BuilderGoldenTensor
        Input tensor
    **kwargs : dict
        Keyword arguments including 'dim', 'begin', 'end', 'step'

    Returns
    -------
    BuilderGoldenTensor
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


def slice_golden(input_tensor: BuilderGoldenTensor, **kwargs) -> BuilderGoldenTensor:
    """
    Golden function for slice operation with TTIR parameter names.

    Parameters
    ----------
    input_tensor : BuilderGoldenTensor
        Input tensor
    **kwargs : dict
        Keyword arguments including slice attributes as MLIR attributes

    Returns
    -------
    BuilderGoldenTensor
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

    shard_map = {}
    for device_id, shard in input_tensor.shard_map.items():
        shard_map[device_id] = shard[slices]

    return BuilderGoldenTensor(shard_map, input_tensor.mesh_shape)


def zeros_golden(**kwargs) -> BuilderGoldenTensor:
    """
    Golden function for zeros operation with TTIR parameter names.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments including 'size'

    Returns
    -------
    BuilderGoldenTensor
        Zero tensor
    """
    size = kwargs.get("shape", [1])
    return BuilderGoldenTensor({0: torch.zeros(size)}, (1, 1))


def ones_golden(**kwargs) -> BuilderGoldenTensor:
    """
    Golden function for ones operation with TTIR parameter names.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments including 'size'

    Returns
    -------
    BuilderGoldenTensor
        Ones tensor
    """
    size = kwargs.get("shape", [1])
    return BuilderGoldenTensor({0: torch.ones(size)}, (1, 1))


def constant_golden(**kwargs) -> BuilderGoldenTensor:
    """
    Golden function for constant operation with TTIR parameter names.

    Parameters
    ----------
    **kwargs : dict
        Keyword arguments including 'value'

    Returns
    -------
    BuilderGoldenTensor
        Constant tensor
    """
    value = kwargs.get("value", [1])
    return BuilderGoldenTensor({0: value}, (1, 1))


def reverse_golden(input_tensor: BuilderGoldenTensor, **kwargs) -> BuilderGoldenTensor:
    """
    Golden function for reverse operation with TTIR parameter names.

    Parameters
    ----------
    input_tensor : BuilderGoldenTensor
        Input tensor
    **kwargs : dict
        Keyword arguments including 'dims'

    Returns
    -------
    BuilderGoldenTensor
        Reversed tensor
    """
    dims = kwargs.get("dimensions", [0])
    return torch.flip(input_tensor, dims)


def arange_golden(
    single_dim_tensor: BuilderGoldenTensor, **kwargs
) -> BuilderGoldenTensor:
    """
    Golden function for arange operation using TTIR kwargs.

    Expected kwargs from builder (ttir_kwargs):
    - start: int
    - end: int
    - step: int
    - arange_dimension: int (ignored here; layout handled by builder output shape)
    """
    start = kwargs.get("start", 0)
    end = kwargs.get("end", 0)
    step = kwargs.get("step", 1)
    output_shards = {}
    for device_id, shard in single_dim_tensor.shard_map.items():
        output_shards[device_id] = torch.arange(
            start=start, end=end, step=step, dtype=torch.float32
        )
    return BuilderGoldenTensor(output_shards, single_dim_tensor.mesh_shape)


def cumsum_golden(input_tensor: BuilderGoldenTensor, **kwargs) -> BuilderGoldenTensor:
    """
    Golden function for cumsum operation with TTIR parameter names.

    Parameters
    ----------
    input_tensor : BuilderGoldenTensor
        Input tensor
    **kwargs : dict
        Keyword arguments containing:
        - dim: int - Dimension along which to compute cumulative sum

    Returns
    -------
    BuilderGoldenTensor
        Cumulative sum of input tensor along specified dimension
    """
    dim = kwargs.get("dim", 0)  # Use the dim parameter from ttir_kwargs
    return torch.cumsum(input_tensor, dim=dim)


def repeat_interleave_golden(
    input_tensor: BuilderGoldenTensor, **kwargs
) -> BuilderGoldenTensor:
    """
    Golden function for repeat_interleave operation with TTIR parameter names.

    Parameters
    ----------
    input_tensor : BuilderGoldenTensor
        Input tensor
    **kwargs : dict
        Keyword arguments including 'repeats' and 'dim'

    Returns
    -------
    BuilderGoldenTensor
        Repeated tensor
    """
    repeats = kwargs.get("repeats", 1)
    dim = kwargs.get("dim", 0)
    return torch.repeat_interleave(input_tensor, repeats, dim=dim)


def _sharding(
    tensor: BuilderGoldenTensor,
    mesh_shape: Tuple[int],
    shard_dims: Tuple[Union[int, None]],
) -> BuilderGoldenTensor:
    assert len(mesh_shape) == len(
        shard_dims
    ), "mesh_shape and shard_dims must have the same length"
    assert len(tensor.shard_map) == 1, "Input tensor must have a single shard"

    shards = [tensor.shard_at(0).clone()]
    for dim_size, shard_dim in zip(mesh_shape, shard_dims):
        temp_shards = []
        if shard_dim is None or shard_dim == -1:
            for shard in shards:
                temp_shards.extend([shard.clone() for _ in range(dim_size)])
        else:
            for shard in shards:
                temp_shards.extend(torch.chunk(shard, dim_size, dim=shard_dim))
        shards = temp_shards

    shard_dictionary = {i: shard for i, shard in enumerate(shards)}
    return BuilderGoldenTensor(shard_dictionary, mesh_shape)


def _unsharding(
    tensor: BuilderGoldenTensor,
    mesh_shape: Tuple[int],
    shard_dims: Tuple[Union[int, None]],
) -> BuilderGoldenTensor:
    assert len(mesh_shape) == len(
        shard_dims
    ), "mesh_shape and shard_dims must have the same length"
    assert len(tensor.shard_map) != 1, "Input tensor must have multiple shards"

    shards = [tensor.shard_at(i).clone() for i in range(len(tensor.shard_map))]
    for dim_size, shard_dim in zip(reversed(mesh_shape), reversed(shard_dims)):
        if shard_dim is None or shard_dim == -1:
            shards = shards[::dim_size]
        else:
            temp_shards = []
            for i in range(0, len(shards), dim_size):
                concat_shard = torch.cat(shards[i : i + dim_size], dim=shard_dim)
                temp_shards.append(concat_shard)
            shards = temp_shards

    return BuilderGoldenTensor({0: shards[0]}, mesh_shape)


def mesh_shard_golden(
    input: BuilderGoldenTensor,
    mesh_shape: Tuple[int, int],
    shard_type: Attribute,
    shard_direction: Attribute,
    shard_shape: Tuple[int, int],
    shard_dims: List[int],
) -> BuilderGoldenTensor:
    """
    Return a tensor which was sharded or unsharded by mesh_shard.

    Parameters
    ----------
    input : BuilderGoldenTensor
        Input tensor to be sharded or unsharded
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
    BuilderGoldenTensor
        Golden tensor which was sharded or unsharded by mesh_shard.
    """

    shard_direction_str = str(shard_direction).lower()
    shard_type_str = str(shard_type).lower()
    if "full_to_shard" in shard_direction_str:
        if "replicate" in shard_type_str:
            shard_dims = [None] * len(mesh_shape)
        return _sharding(input, mesh_shape, shard_dims)
    elif "shard_to_full" in shard_direction_str:
        if "replicate" in shard_type_str:
            return _unsharding(input, [1], [1])
        else:
            return _unsharding(input, mesh_shape, shard_dims)


def all_gather_golden(
    input: BuilderGoldenTensor,
    all_gather_dim: int,
    cluster_axis: int,
) -> BuilderGoldenTensor:
    """
    Return a BuilderGoldenTensor which was gathered from all devices.

    Parameters
    ----------
    input : BuilderGoldenTensor
        Input tensor to gather from all devices
    all_gather_dim : int
        Dimension to gather along
    cluster_axis : int
        Axis of the cluster for gathering

    Returns
    -------
    BuilderGoldenTensor
        BuilderGoldenTensor which was gathered from all devices
    """

    output_shards = [None] * len(input.shard_map)
    grouped_shards = input.group_by_axis(cluster_axis)
    for group in grouped_shards:
        gathered_tensor = torch.cat(list(group.values()), dim=all_gather_dim)
        for id in group.keys():
            output_shards[id] = gathered_tensor.clone()
    return BuilderGoldenTensor(
        {i: t for i, t in enumerate(output_shards)}, input.mesh_shape
    )


# Map of supported reduction keywords to callable functions
_REDUCE = {
    "sum": lambda xs: torch.sum(torch.stack(xs), 0),
    "mean": lambda xs: torch.mean(torch.stack(xs), 0),
    "max": lambda xs: torch.amax(torch.stack(xs), 0),
    "min": lambda xs: torch.amin(torch.stack(xs), 0),
    "std": lambda xs: torch.std(torch.stack(xs), 0),  # default correction=1
    "var": lambda xs: torch.var(torch.stack(xs), 0),
}


def _reduce(inputs: List[torch.Tensor], reduce_type: Attribute) -> BuilderGoldenTensor:
    key = str(reduce_type).lower()
    # Handle alias form like "reduce_type<sum>"
    if key.startswith("#ttcore.reduce_type<") and key.endswith(">"):
        key = key[20:-1]
    try:
        return _REDUCE[key](inputs)
    except KeyError as err:
        raise ValueError(f"Unsupported reduce type: {reduce_type}") from err


def all_reduce_golden(
    input: BuilderGoldenTensor,
    cluster_axis: int,
    reduce_type: Attribute,
) -> BuilderGoldenTensor:
    """
    Return a BuilderGoldenTensor which was reduced across devices.

    Parameters
    ----------
    input : BuilderGoldenTensor
        Input tensor to reduce across devices
    cluster_axis : int
        Axis of the cluster for reduction
    reduce_type : Attribute
        Type of reduction operation

    Returns
    -------
    BuilderGoldenTensor
        BuilderGoldenTensor which was reduced across devices
    """

    output_shards = [None] * len(input.shard_map)
    grouped_shards = input.group_by_axis(cluster_axis)
    for group in grouped_shards:
        group_tensors = list(group.values())
        reduced_tensor = _reduce(group_tensors, reduce_type)
        for id in group.keys():
            output_shards[id] = reduced_tensor.clone()
    return BuilderGoldenTensor(
        {i: t for i, t in enumerate(output_shards)}, input.mesh_shape
    )


def reduce_scatter_golden(
    input: BuilderGoldenTensor,
    reduce_type: Attribute,
    scatter_dim: int,
    cluster_axis: int,
) -> BuilderGoldenTensor:
    """
    Return a BuilderGoldenTensor which was reduced and scattered across devices.

    Parameters
    ----------
    input : BuilderGoldenTensor
        Input tensor to reduce and scatter
    reduce_type : Attribute
        Type of reduction operation
    scatter_dim : int
        Dimension to scatter along
    cluster_axis : int
        Axis of the cluster for operation

    Returns
    -------
    BuilderGoldenTensor
        BuilderGoldenTensor which was reduced and scattered across devices
    """

    output_shards = [None] * len(input.shard_map)
    grouped_shards = input.group_by_axis(cluster_axis)
    for group in grouped_shards:
        group_tensors = list(group.values())
        reduced_tensor = _reduce(group_tensors, reduce_type)
        scattered_tensor = torch.chunk(reduced_tensor, len(group), dim=scatter_dim)
        for index, id in enumerate(group.keys()):
            output_shards[id] = scattered_tensor[index].clone()
    return BuilderGoldenTensor(
        {i: t for i, t in enumerate(output_shards)}, input.mesh_shape
    )


def collective_permute_golden(
    input: BuilderGoldenTensor,
    source_target_pairs: List[Tuple[int, int]],
) -> BuilderGoldenTensor:
    """
    Return a BuilderGoldenTensor which was permuted across devices.

    Parameters
    ----------
    input : BuilderGoldenTensor
        Input tensor to permute across devices
    source_target_pairs : List[Tuple[int, int]]
        List of (source, target) device ID pairs for permutation

    Returns
    -------
    BuilderGoldenTensor
        BuilderGoldenTensor which was permuted across devices
    """

    output_shards = [torch.zeros_like(shard) for shard in input.shard_map.values()]
    for src, tgt in source_target_pairs:
        output_shards[tgt] = input.shard_at(src).clone()
    return BuilderGoldenTensor(
        {i: t for i, t in enumerate(output_shards)}, input.mesh_shape
    )


def all_to_all_golden(
    input: BuilderGoldenTensor,
    split_dim: int,
    concat_dim: int,
    split_count: int,
    replica_groups: List[List[int]],
) -> BuilderGoldenTensor:
    """
    Return a BuilderGoldenTensor which was redistributed across devices.

    Parameters
    ----------
    input : BuilderGoldenTensor
        Input tensor to perform all-to-all communication on
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
    BuilderGoldenTensor
        BuilderGoldenTensor which was redistributed across devices.
    """

    output_shards = [None] * len(input.shard_map)
    for group in replica_groups:
        assert len(group) == split_count, "group size must equal split_count"
        splits_per_src: List[Tuple[torch.Tensor, ...]] = [
            torch.chunk(input.shard_at(dev_id), split_count, dim=split_dim)
            for dev_id in group
        ]
        for dst_idx in range(split_count):
            output_shards[group[dst_idx]] = torch.cat(
                [splits_per_src[src_idx][dst_idx] for src_idx in range(split_count)],
                dim=concat_dim,
            )
    return BuilderGoldenTensor(
        {i: t for i, t in enumerate(output_shards)}, input.mesh_shape
    )


def collective_broadcast_golden(
    input: BuilderGoldenTensor,
    replica_groups: List[Tuple[int, int]],
) -> BuilderGoldenTensor:
    """
    Return a BuilderGoldenTensor which was broadcasted across devices.

    Parameters
    ----------
    input : BuilderGoldenTensor
        Input tensor to broadcast across devices
    replica_groups : List[Tuple[int, int]]
        Groups of replica devices for broadcasting

    Returns
    -------
    BuilderGoldenTensor
        BuilderGoldenTensor which was broadcasted across devices.
    """

    output_shards = [None] * len(input.shard_map)
    for group in replica_groups:
        for device in group:
            output_shards[device] = input.shard_at(group[0]).clone()
    return BuilderGoldenTensor(
        {i: t for i, t in enumerate(output_shards)}, input.mesh_shape
    )


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
    if (
        ttir_op_class == ttir.ToLayoutOp or ttir_op_class == d2m.ToLayoutOp
    ) and "tilize" in kwargs:
        if kwargs["tilize"]:
            return tilize_golden
        else:
            return untilize_golden

    if ttir_op_class in GOLDEN_MAPPINGS:
        return GOLDEN_MAPPINGS[ttir_op_class]

    return None


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
    ttir.ErfOp: torch.erf,
    ttir.ErfcOp: torch.erfc,
    ttir.FloorOp: torch.floor,
    ttir.GeluOp: torch.nn.functional.gelu,
    ttir.IsFiniteOp: torch.isfinite,
    ttir.NegOp: torch.neg,
    ttir.TanOp: torch.tan,
    ttir.AtanOp: torch.atan,
    ttir.TanhOp: torch.tanh,
    ttir.ReciprocalOp: torch.reciprocal,
    ttir.ReluOp: torch.relu,
    ttir.Relu6Op: torch.nn.functional.relu6,
    ttir.RsqrtOp: torch.rsqrt,
    ttir.SigmoidOp: torch.sigmoid,
    ttir.SignOp: torch.sign,
    ttir.SiluOp: silu_golden,
    ttir.SinOp: torch.sin,
    ttir.SqrtOp: torch.sqrt,
    ttir.LogOp: torch.log,
    ttir.Log1pOp: torch.log1p,
    ttir.Expm1Op: torch.expm1,
    ttir.ExpOp: torch.exp,
    # Elementwise binary operations
    ttir.AddOp: torch.add,
    ttir.Atan2Op: torch.atan2,
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
    ttir.LogicalLeftShiftOp: logical_left_shift_golden,
    ttir.LogicalOrOp: logical_or_golden,
    ttir.LogicalRightShiftOp: logical_right_shift_golden,
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
    ttir.SortOp: sort_golden,
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
    ttir.ClampTensorOp: clamp_tensor_golden,
    ttir.CumSumOp: cumsum_golden,
    ttir.BroadcastOp: torch.broadcast_to,
    ttir.PadOp: pad_golden,
    ttir.IndexSelectOp: select_golden,
    ttir.IndexOp: index_golden,
    ttir.SliceStaticOp: slice_golden,
    ttir.GatherOp: gather_golden,
    # Neural network operations
    ttir.SoftmaxOp: softmax_golden,
    ttir.MatmulOp: torch.matmul,
    ttir.EmbeddingOp: embedding_golden,
    ttir.Upsample2dOp: upsample2d_golden,
    ttir.BatchNormInferenceOp: batch_norm_golden,
    ttir.RMSNormOp: rms_norm_golden,
    # Type operations
    ttir.TypecastOp: typecast_golden,
    # Tensor creation
    ttir.ZerosOp: zeros_golden,
    ttir.OnesOp: ones_golden,
    ttir.ConstantOp: constant_golden,
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
    ttir.AvgPool2dOp: avg_pool2d_golden,
    ttir.ArgMaxOp: argmax_golden,
    ttir.LinearOp: linear_golden,
    ttir.DotGeneralOp: dot_general_golden,
    # Layout operations (identity functions) — accept and ignore extra kwargs like reinterpretLayout
    ttir.ToLayoutOp: (lambda x, **kwargs: x),
    # D2M Layout operations (identity functions)
    d2m.ToLayoutOp: (lambda x, **kwargs: x),
    d2m.ViewLayoutOp: (lambda x, **kwargs: x),
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
    # StableHLO elementwise operations
    stablehlo.AddOp: torch.add,
    stablehlo.AbsOp: torch.abs,
    stablehlo.CeilOp: torch.ceil,
    stablehlo.CosineOp: torch.cos,
    stablehlo.ExpOp: torch.exp,
    stablehlo.FloorOp: torch.floor,
    stablehlo.LogOp: torch.log,
    stablehlo.LogisticOp: torch.sigmoid,
    stablehlo.NegOp: torch.neg,
    stablehlo.RsqrtOp: torch.rsqrt,
    stablehlo.SineOp: torch.sin,
    stablehlo.SqrtOp: torch.sqrt,
    stablehlo.TanOp: torch.tan,
    # TTNN elementwise operations
    ttnn.MultiplyOp: torch.multiply,
}
