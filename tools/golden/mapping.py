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
import einops
import torch
import torch.nn.functional
from ttmlir.dialects import ttir, stablehlo, d2m, ttnn, ttcore, sdy, debug
from ttmlir.ir import *
from ttmlir.passes import DataType


class GoldenMapTensor:
    """
    GoldenMapTensor is a utility class for managing golden values for single device or multi device tensors.
    GoldenMapTensor represents a list of torch.Tensor objects, each representing a shard of a tensor.
    For single device tensors, it contains a single shard.

    How is this class compatible with torch.* operations?
      * For read-only tensor attributes (like shape, dtype, device, etc.), GoldenMapTensor forwards attribute access to the first shard.
      * For torch.* operations, the class implements the `__torch_function__` protocol.
      * When a torch function is called on a GoldenMapTensor, the function is applied independently to each shard.
      * The results are collected into a new GoldenMapTensor.
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

    # Mutating methods. We will always return a new GoldenMapTensor to avoid in-place mutations by design.
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

    def _get_runtime_compatible_torch_dtype(self, dtype: torch.dtype) -> torch.dtype:
        compatible_dtypes = [
            torch.float16,
            torch.bfloat16,
            torch.uint8,
            torch.uint16,
            torch.uint32,
            torch.int32,
            torch.float32,
        ]

        if dtype in compatible_dtypes:
            return dtype
        elif dtype in [torch.qint32, torch.int64]:
            return torch.int32
        elif dtype == torch.bool:
            return torch.bfloat16
        else:
            return torch.float32

    def golden_map_tensor_as_torch_tensors(self) -> Dict[int, torch.Tensor]:
        """
        Return shard tensors as plain torch.Tensor per device, ensuring:
          - each shard is contiguous in memory
          - quantized tensors are converted to their integer representation (int_repr)
          - int64 shards are downcast to int32 for compatibility with borrowed tensor creation.
        """
        torch_goldens: Dict[int, torch.Tensor] = dict(self.contiguous().shard_map)
        for device_id, torch_golden in torch_goldens.items():
            dtype = self._get_runtime_compatible_torch_dtype(torch_golden.dtype)
            if getattr(torch_golden, "is_quantized", False):
                # For quantized tensors, use the underlying integer representation
                torch_goldens[device_id] = torch_golden.int_repr()
            else:
                torch_goldens[device_id] = torch_golden.to(dtype)
        return torch_goldens

    # ----- Private static methods -----
    def __getitem__(self, key: int) -> GoldenMapTensor:
        out_shards = {k: v.__getitem__(key) for k, v in self._shard_map.items()}
        ref = next(iter(out_shards.values()))
        if not all(isinstance(t, torch.Tensor) for t in out_shards.values()):
            return out_shards
        # Wrap
        return GoldenMapTensor(out_shards, self.mesh_shape)

    def _binary_map(self, other, op):
        if isinstance(other, GoldenMapTensor):
            keys = sorted(self._shard_map.keys())
            if set(keys) != set(other._shard_map.keys()):
                raise RuntimeError("Shard key mismatch between operands.")
            out_shards = {k: op(self._shard_map[k], other._shard_map[k]) for k in keys}
        else:
            out_shards = {k: op(t, other) for k, t in self._shard_map.items()}
        # Always wrap (even 0-D)
        return GoldenMapTensor(out_shards, self._mesh_shape)

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
                    yield from GoldenMapTensor._walk_tree(v)
            elif isinstance(tree, dict):
                for v in tree.values():
                    yield from GoldenMapTensor._walk_tree(v)
            else:
                yield tree

    # ----- Private methods -----

    def __getattr__(self, name: str) -> Any:
        if name in GoldenMapTensor._SAFE_TENSOR_ATTRS:
            return getattr(self._shard_map[0], name)
        elif name in GoldenMapTensor._MUTATING_METHODS:

            def method(*args, **kwargs):
                func = GoldenMapTensor._MUTATING_METHODS[name]
                return self.apply_shardwise(lambda shard: func(shard, *args, **kwargs))

            return method
        raise AttributeError(
            f"'GoldenMapTensor' has no attribute '{name}'. "
            "For mutating ops call torch.* directly."
        )

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if not any(issubclass(t, cls) for t in types):
            return NotImplemented

        # Collect all GoldenMapTensor inputs.
        st_inputs = [
            a for a in GoldenMapTensor._walk_tree(args, kwargs) if isinstance(a, cls)
        ]
        shard_counts = {len(st.shard_map) for st in st_inputs}
        if len(shard_counts) != 1:
            raise RuntimeError("All GoldenMapTensors must have the same shard count.")

        # All shard_maps should share the same set of keys.
        shard_keys = [set(st.shard_map.keys()) for st in st_inputs]
        if not all(keys == shard_keys[0] for keys in shard_keys[1:]):
            raise RuntimeError(
                "All GoldenMapTensors must have the same shard keys (devices amongst which the GoldenMapTensor lives)."
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

        # If all results are Tensors and compatible, wrap back into GoldenMapTensor.
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

    def zeros_like_builder(self, shape) -> GoldenMapTensor:
        shard_map = {}
        for device_id, shard in self.shard_map.items():
            shard_map[device_id] = torch.zeros(shape, dtype=shard.dtype)
        return GoldenMapTensor(shard_map, self.mesh_shape)

    def apply_shardwise(
        input_tensor: GoldenMapTensor, func: Callable
    ) -> GoldenMapTensor:
        shard_map = {}
        for device_id, shard in input_tensor.shard_map.items():
            output_shard = shard.clone()
            shard_map[device_id] = func(output_shard)
        return GoldenMapTensor(shard_map, input_tensor.mesh_shape)

    def shard_at(self, device_id: int) -> GoldenMapTensor:
        if device_id not in self._shard_map:
            raise KeyError(f"Device ID {device_id} not found in shard map.")
        return self._shard_map[device_id]

    def clone(self) -> GoldenMapTensor:
        shard_map = {
            device_id: shard.clone() for device_id, shard in self.shard_map.items()
        }
        return GoldenMapTensor(shard_map, self.mesh_shape)

    def contiguous(self) -> GoldenMapTensor:
        return GoldenMapTensor(
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
    if isinstance(attr, IntegerAttr):
        return attr.value
    if isinstance(attr, BoolAttr):
        return attr.value
    if isinstance(attr, (DenseI64ArrayAttr, DenseI32ArrayAttr, DenseBoolArrayAttr)):
        return list(attr)
    if isinstance(attr, ArrayAttr):
        return [unpack_mlir_attr(item) for item in attr]
    if isinstance(attr, (list, tuple)):
        return list(attr)
    if isinstance(attr, (int, bool, float)):
        return attr
    if isinstance(attr, FloatAttr):
        return attr.value
    if isinstance(attr, StringAttr):
        return attr.value
    if isinstance(attr, DenseElementsAttr):
        array = np.array(attr)
        return array
    raise ValueError(f"Unexpected attribute type: {type(attr)}")


def mlir_type_to_torch_dtype(mlir_type: Type) -> torch.dtype:
    type_str = str(mlir_type)

    if isinstance(mlir_type, BF16Type) or type_str == "bf16":
        return torch.bfloat16
    elif isinstance(mlir_type, F16Type) or type_str == "f16":
        return torch.float16
    elif isinstance(mlir_type, F32Type) or type_str == "f32":
        return torch.float32
    elif isinstance(mlir_type, F64Type) or type_str == "f64":
        return torch.float64
    elif isinstance(mlir_type, IntegerType):
        width = mlir_type.width
        is_signed = mlir_type.is_signed
        is_unsigned = mlir_type.is_unsigned

        if width == 1:
            return torch.bool
        elif width == 8:
            if is_unsigned:
                return torch.uint8
            else:
                return torch.int8
        elif width == 16:
            if is_unsigned:
                return torch.uint16
            else:
                return torch.int16
        elif width == 32:
            if is_unsigned:
                return torch.uint32
            else:
                return torch.int32
        elif width == 64:
            if is_unsigned:
                return torch.uint64
            else:
                return torch.int64
        else:
            raise TypeError(f"Unsupported integer width: {width}")
    else:
        raise TypeError(f"Unsupported MLIR type: {mlir_type}")


def mlir_datatype_to_torch_dtype(mlir_datatype: DataType) -> torch.dtype:

    match str(mlir_datatype):
        case "DataType.Float16":
            return torch.float16
        case "DataType.BFloat16":
            return torch.bfloat16
        case "DataType.UInt8":
            return torch.uint8
        case "DataType.UInt16":
            return torch.uint16
        case "DataType.UInt32":
            return torch.uint32
        case "DataType.Int32":
            return torch.int32
        case "DataType.Float32":
            return torch.float32
        case _:
            raise TypeError(f"Unsupported MLIR DataType: {mlir_datatype}")


def cbrt_golden(x: GoldenMapTensor) -> GoldenMapTensor:
    """
    Custom golden function for cubic root.

    Parameters
    ----------
    x : GoldenMapTensor
        Input tensor

    Returns
    -------
    GoldenMapTensor
        GoldenMapTensor containing the cubic root of each element in the input tensor
    """
    golden_sign = torch.sign(x)
    golden_cbrt = torch.pow(torch.abs(x), 1 / 3)
    return torch.mul(golden_sign, golden_cbrt)


def conv2d_golden(
    input_tensor: GoldenMapTensor,
    weight: GoldenMapTensor,
    bias: Optional[GoldenMapTensor],
    stride: Union[IntegerAttr, DenseI32ArrayAttr],
    padding: Union[IntegerAttr, DenseI32ArrayAttr],
    dilation: Union[IntegerAttr, DenseI32ArrayAttr],
    groups: IntegerAttr,
    batch_dim: IntegerAttr,
    height_dim: IntegerAttr,
    width_dim: IntegerAttr,
    channel_dim: IntegerAttr,
) -> GoldenMapTensor:
    """
    Custom golden function for conv2d with layout transformation.

    Parameters
    ----------
    input_tensor : GoldenMapTensor
        Input tensor for convolution
    weight : GoldenMapTensor
        Convolution weight tensor
    bias : Optional[GoldenMapTensor]
        Optional bias tensor
    stride : Union[IntegerAttr, DenseI32ArrayAttr]
        Stride for convolution
    padding : Union[IntegerAttr, DenseI32ArrayAttr]
        Padding for convolution
    dilation : Union[IntegerAttr, DenseI32ArrayAttr]
        Dilation for convolution
    groups : IntegerAttr
        Number of groups for grouped convolution
    batch_dim : IntegerAttr
        Batch dimension index
    height_dim : IntegerAttr
        Height dimension index
    width_dim : IntegerAttr
        Width dimension index
    channel_dim : IntegerAttr
        Channel dimension index

    Returns
    -------
    GoldenMapTensor
        Result of 2D convolution with layout transformation
    """
    # ttir can handle a broadcastable bias in the shape [1, 1, 1, C_out], but PyTorch requires the bias to be rank 1: [C_out].
    if bias is not None:
        bias = bias.squeeze()  # Removes all dims of size 1

    stride = unpack_mlir_attr(stride)
    padding = unpack_mlir_attr(padding)
    dilation = unpack_mlir_attr(dilation)
    groups = unpack_mlir_attr(groups)

    batch_dim = unpack_mlir_attr(batch_dim)
    height_dim = unpack_mlir_attr(height_dim)
    width_dim = unpack_mlir_attr(width_dim)
    channel_dim = unpack_mlir_attr(channel_dim)

    # Compute permutation to convert any layout to NCHW (batch=0, channel=1, height=2, width=3).
    to_nchw_perm = [batch_dim, channel_dim, height_dim, width_dim]

    is_nchw = to_nchw_perm == [0, 1, 2, 3]

    copied_input_tensor = input_tensor.clone()
    if not is_nchw:
        copied_input_tensor = copied_input_tensor.permute(to_nchw_perm)

    # Handle padding format - TTIR uses [low_h, high_h, low_w, high_w] but PyTorch expects [pad_h, pad_w].
    asymmetric_padding = False
    if isinstance(padding, (list, tuple)) and len(padding) == 4:
        low_h, high_h, low_w, high_w = [int(p) for p in padding]
        if low_h == high_h and low_w == high_w:
            padding = [low_h, low_w]
        else:
            copied_input_tensor = torch.nn.functional.pad(
                copied_input_tensor,
                [low_w, high_w, low_h, high_h],
                mode="constant",
                value=0,
            )
            asymmetric_padding = True
            padding = [0, 0]

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
    # Permute output back to original layout if we permuted the input.
    if not is_nchw:
        from_nchw_perm = [0] * 4
        for i, p in enumerate(to_nchw_perm):
            from_nchw_perm[p] = i
        result = result.permute(from_nchw_perm)
    return result


def conv_transpose2d_golden(
    input_tensor: GoldenMapTensor,
    weight: GoldenMapTensor,
    bias: Optional[GoldenMapTensor],
    stride: Union[IntegerAttr, DenseI32ArrayAttr],
    padding: Union[IntegerAttr, DenseI32ArrayAttr],
    output_padding: Union[IntegerAttr, DenseI32ArrayAttr],
    dilation: Union[IntegerAttr, DenseI32ArrayAttr],
    groups: IntegerAttr,
    batch_dim: IntegerAttr,
    height_dim: IntegerAttr,
    width_dim: IntegerAttr,
    channel_dim: IntegerAttr,
) -> GoldenMapTensor:
    """
    Custom golden function for conv_transpose2d with layout transformation.

    Parameters
    ----------
    input_tensor : GoldenMapTensor
        Input tensor for transposed convolution
    weight : GoldenMapTensor
        Convolution weight tensor
    bias : Optional[GoldenMapTensor]
        Optional bias tensor
    stride : Union[IntegerAttr, DenseI32ArrayAttr]
        Stride for transposed convolution
    padding : Union[IntegerAttr, DenseI32ArrayAttr]
        Padding for transposed convolution
    output_padding : Union[IntegerAttr, DenseI32ArrayAttr]
        Additional size added to output shape
    dilation : Union[IntegerAttr, DenseI32ArrayAttr]
        Dilation of the kernel
    groups : IntegerAttr
        Number of blocked connections from input to output channels
    batch_dim : IntegerAttr
        Batch dimension index
    height_dim : IntegerAttr
        Height dimension index
    width_dim : IntegerAttr
        Width dimension index
    channel_dim : IntegerAttr
        Channel dimension index

    Returns
    -------
    GoldenMapTensor
        Result of 2D transposed convolution with layout transformation
    """
    stride = unpack_mlir_attr(stride)
    padding = unpack_mlir_attr(padding)
    output_padding = unpack_mlir_attr(output_padding)
    dilation = unpack_mlir_attr(dilation)
    groups = unpack_mlir_attr(groups)

    batch_dim = unpack_mlir_attr(batch_dim)
    height_dim = unpack_mlir_attr(height_dim)
    width_dim = unpack_mlir_attr(width_dim)
    channel_dim = unpack_mlir_attr(channel_dim)

    if bias is not None:
        bias = bias.squeeze()

    # Compute permutation to convert any layout to NCHW (batch=0, channel=1, height=2, width=3).
    to_nchw_perm = [batch_dim, channel_dim, height_dim, width_dim]

    is_nchw = to_nchw_perm == [0, 1, 2, 3]

    copied_input_tensor = input_tensor.clone()
    if not is_nchw:
        copied_input_tensor = copied_input_tensor.permute(to_nchw_perm)

    result = torch.nn.functional.conv_transpose2d(
        copied_input_tensor,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
    )

    # Permute output back to original layout if we permuted the input.
    if not is_nchw:
        from_nchw_perm = [0] * 4
        for i, p in enumerate(to_nchw_perm):
            from_nchw_perm[p] = i
        result = result.permute(from_nchw_perm)
    return result


def ttir_max_pool2d_golden(
    input_tensor: GoldenMapTensor,
    kernel_attr: Union[IntegerAttr, DenseI32ArrayAttr],
    stride_attr: Union[IntegerAttr, DenseI32ArrayAttr],
    padding_attr: Union[IntegerAttr, DenseI32ArrayAttr],
    dilation_attr: Union[IntegerAttr, DenseI32ArrayAttr],
    ceil_mode_attr: BoolAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    """
    Custom golden function for max_pool2d with layout transformation.

    Parameters
    ----------
    input_tensor : GoldenMapTensor
        Input tensor for max pooling
    kernel_attr : Union[IntegerAttr, DenseI32ArrayAttr]
        Size of the pooling kernel
    stride_attr : Union[IntegerAttr, DenseI32ArrayAttr]
        Stride for pooling operation
    padding_attr : Union[IntegerAttr, DenseI32ArrayAttr]
        Padding for pooling operation
    dilation_attr : Union[IntegerAttr, DenseI32ArrayAttr]
        Dilation for pooling operation
    ceil_mode_attr : BoolAttr
        Whether to use ceiling mode for pooling
    output_type_mlir : Type
        MLIR type for the output tensor

    Returns
    -------
    GoldenMapTensor
        Result of 2D max pooling with layout transformation
    """
    kernel_size = unpack_mlir_attr(kernel_attr)
    stride = unpack_mlir_attr(stride_attr)
    padding = unpack_mlir_attr(padding_attr)
    dilation = unpack_mlir_attr(dilation_attr)
    ceil_mode = unpack_mlir_attr(ceil_mode_attr)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)

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
        kernel_size=kernel_size,
        stride=stride,
        padding=torch_padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )
    input_tensor = input_tensor.transpose(-2, -1).transpose(-3, -2)
    result = maxpool_object(input_tensor)
    result = result.transpose(-3, -2).transpose(-2, -1)
    return result.to(output_dtype)


def avg_pool2d_golden(input_tensor: GoldenMapTensor, **kwargs) -> GoldenMapTensor:
    """
    Custom golden function for avg_pool2d with layout transformation.

    Parameters
    ----------
    input_tensor : GoldenMapTensor
        Input tensor for avg pooling
    **kwargs : dict
        Keyword arguments containing:
        - kernel_size: Union[int, List[int]] - Size of the pooling kernel
        - stride: Union[int, List[int]] - Stride for pooling operation
        - padding: Union[int, List[int]] - Padding for pooling operation
        - ceil_mode: bool - Whether to use ceiling mode for pooling
        - count_include_pad: bool - Whether to include padding in the average calculation

    Returns
    -------
    GoldenMapTensor
        Result of 2D avg pooling with layout transformation
    """
    # Get parameters from ttir_kwargs
    kernel_size = kwargs.get("kernel")
    stride = kwargs.get("stride", kernel_size)  # Default stride = kernel size
    padding = kwargs.get("padding", 0)
    dilation = kwargs.get("dilation", 1)  # Default dilation = 1
    ceil_mode = kwargs.get("ceil_mode", False)
    count_include_pad = kwargs.get("count_include_pad", True)

    kernel_size = unpack_mlir_attr(kernel_size)
    stride = unpack_mlir_attr(stride)
    padding = unpack_mlir_attr(padding)
    dilation = unpack_mlir_attr(dilation)

    # Check if padding exceeds half kernel size (tt-metal constraint)
    # This mirrors the decomposition in TTIRToTTNN.cpp
    if isinstance(kernel_size, (list, tuple)):
        kernel_h, kernel_w = kernel_size
    else:
        kernel_h = kernel_w = kernel_size

    max_pad_h = kernel_h // 2
    max_pad_w = kernel_w // 2

    # Convert padding from [top, left, bottom, right] format or other formats
    if isinstance(padding, (list, tuple)) and len(padding) == 4:
        top, left, bottom, right = padding
    elif isinstance(padding, (list, tuple)) and len(padding) == 2:
        top = bottom = padding[0]
        left = right = padding[1]
    elif isinstance(padding, int):
        top = bottom = left = right = padding
    else:
        top = bottom = left = right = 0

    # TTIR avg_pool2d is channels last. PyTorch avg_pool2d is channels first.
    # Convert to channels first before any padding operations
    input_tensor = input_tensor.transpose(-2, -1).transpose(-3, -2)

    # If padding exceeds half kernel size, we need to manually pad first
    if top > max_pad_h or left > max_pad_w or bottom > max_pad_h or right > max_pad_w:
        import torch.nn.functional as F

        # Manually apply padding with zeros for avg pooling
        # For channels-first (N, C, H, W), F.pad expects [left, right, top, bottom]
        manual_padding = [left, right, top, bottom]
        input_tensor = F.pad(input_tensor, manual_padding, mode="constant", value=0.0)
        # Now use zero padding for the pooling operation
        torch_padding = 0
    else:
        # Standard case: padding within limits
        if top == bottom and left == right:
            torch_padding = (top, left)
        else:
            # For asymmetric padding, we need to manually pad the input tensor first
            import torch.nn.functional as F

            # For channels-first (N, C, H, W), F.pad expects [left, right, top, bottom]
            manual_padding = [left, right, top, bottom]
            input_tensor = F.pad(
                input_tensor, manual_padding, mode="constant", value=0.0
            )
            torch_padding = 0

    avgpool_object = torch.nn.AvgPool2d(
        kernel_size, stride, torch_padding, ceil_mode, count_include_pad
    )
    result = avgpool_object(input_tensor)
    # Convert back to channels last
    result = result.transpose(-3, -2).transpose(-2, -1)
    return result


def global_avg_pool2d_golden(
    input_tensor: GoldenMapTensor,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    """
    Custom golden function for global_avg_pool2d with layout transformation.

    Global average pooling performs average pooling over the entire spatial dimensions.

    Parameters
    ----------
    input_tensor : GoldenMapTensor
        Input tensor for global avg pooling (N, H, W, C format - channels last)
    **kwargs : dict
        Additional keyword arguments (unused for global pooling)

    Returns
    -------
    GoldenMapTensor
        Result of global 2D avg pooling with layout transformation (N, 1, 1, C format)
    """
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    # TTIR global_avg_pool2d is channels last. PyTorch adaptive_avg_pool2d is channels first.
    # Convert from (N, H, W, C) to (N, C, H, W)
    input_tensor = input_tensor.transpose(-2, -1).transpose(-3, -2)

    # Use adaptive average pooling to reduce spatial dimensions to 1x1
    import torch.nn.functional as F

    result = F.adaptive_avg_pool2d(input_tensor, (1, 1))

    # Convert back from (N, C, 1, 1) to (N, 1, 1, C)
    result = result.transpose(-3, -2).transpose(-2, -1)
    return result.to(output_dtype)


def batch_norm_golden(
    input_tensor: GoldenMapTensor,
    scale: GoldenMapTensor,
    offset: GoldenMapTensor,
    mean: GoldenMapTensor,
    variance: GoldenMapTensor,
    epsilon: float = 1e-5,
    training: bool = False,
    dim: int = 1,
) -> GoldenMapTensor:
    """
    Custom golden function for batch normalization with layout transformation.

    Parameters
    ----------
    input_tensor : GoldenMapTensor
        Input tensor for batch normalization
    scale : GoldenMapTensor
        Scale tensor for batch normalization
    offset : GoldenMapTensor
        Offset tensor for batch normalization
    mean : GoldenMapTensor
        Mean tensor for batch normalization
    variance : GoldenMapTensor
        Variance tensor for batch normalization
    epsilon : float, optional
        Small value to avoid division by zero (default: 1e-5)
    training : bool, optional
        Whether the model is in training mode (default: False)
    dim : int, optional
        Dimension to apply batch normalization over (default: 1)

    Returns
    -------
    GoldenMapTensor
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
    input: GoldenMapTensor,
    weight: Optional[GoldenMapTensor] = None,
    bias: Optional[GoldenMapTensor] = None,
    normalized_shape: List[int] = None,
    epsilon: float = 1e-5,
) -> GoldenMapTensor:
    """
    Custom golden function for RMS normalization operation.
    Parameters
    ----------
    input : GoldenMapTensor
        Input tensor to RMS normalization operation
    weight : GoldenMapTensor, optional
        Weight tensor for scaling (default: None)
    bias : GoldenMapTensor, optional
        Bias tensor for shifting (default: None)
    normalized_shape : List[int], optional
        Shape of the input tensor to normalize (default: None)
    epsilon : float, optional
        Small value to avoid division by zero (default: 1e-5)
    Returns
    -------
    GoldenMapTensor
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


def ttir_rms_norm_golden(
    input: GoldenMapTensor,
    weight: Optional[GoldenMapTensor],
    bias: Optional[GoldenMapTensor],
    normalized_shape: ArrayAttr,
    epsilon: FloatAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    normalized_shape = unpack_mlir_attr(normalized_shape)
    epsilon = unpack_mlir_attr(epsilon)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    input_float = input.float()

    rms_norm = torch.nn.functional.rms_norm(
        input_float,
        normalized_shape=normalized_shape,
        weight=weight,
        eps=epsilon,
    )

    if bias is not None:
        rms_norm = torch.add(rms_norm, bias)

    return rms_norm.to(output_dtype)


def ttir_distributed_rms_norm_golden(
    input: GoldenMapTensor,
    weight: Optional[GoldenMapTensor],
    residual: Optional[GoldenMapTensor],
    cluster_axis_attr: IntegerAttr,
    epsilon_attr: FloatAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    """Distributed RMS normalization golden.

    Simulates fused_rms_minimal: for each group of devices along
    ``cluster_axis``, concatenate the per-device shards to compute
    globally-correct RMS statistics, apply RMS norm + weight on the
    full tensor, then chunk the result back so each device gets only
    its local portion.  The per-device output shape equals the
    per-device input shape (only the stats are all-gathered, not
    the data).

    Parameters
    ----------
    input : GoldenMapTensor
        Per-device input tensor shards.
    weight : Optional[GoldenMapTensor]
        Per-device weight shards. If present, applied after normalization.
    residual : Optional[GoldenMapTensor]
        Per-device residual shards. If present, added to input before
        normalization.
    cluster_axis_attr : IntegerAttr
        Mesh axis (0 or 1) along which devices exchange RMS statistics.
    epsilon_attr : FloatAttr
        Small constant added to the variance for numerical stability.
    output_type_mlir : Type
        MLIR element type used to determine the output torch dtype.

    Returns
    -------
    GoldenMapTensor
        Per-device normalized output shards, each with the same shape
        as the corresponding input shard.
    """
    cluster_axis = unpack_mlir_attr(cluster_axis_attr)
    epsilon = unpack_mlir_attr(epsilon_attr)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)

    num_shards = len(input.shard_map)
    output_shards = [None] * num_shards
    grouped_shards = input.group_by_axis(cluster_axis)

    for group in grouped_shards:
        group_ids = list(group.keys())
        group_tensors = [group[id] for id in group_ids]

        # Add residual per-shard if present
        if residual is not None:
            group_tensors = [
                inp + residual.shard_map[id]
                for inp, id in zip(group_tensors, group_ids)
            ]

        # Concatenate shards along last dim to get full tensor for global RMS norm stats
        full_tensor = torch.cat(group_tensors, dim=-1).float()

        # Compute RMS norm on full tensor
        normalized_shape = [full_tensor.shape[-1]]
        weight_tensor = None
        if weight is not None:
            # Concatenate weight shards to get the full weight
            weight_shards = [weight.shard_map[id] for id in group_ids]
            weight_tensor = torch.cat(weight_shards, dim=-1)

        rms_result = torch.nn.functional.rms_norm(
            full_tensor,
            normalized_shape=normalized_shape,
            weight=weight_tensor,
            eps=epsilon,
        )

        rms_result = rms_result.to(output_dtype)

        # Split back into per-device chunks (output shape == input shape).
        per_shard_chunks = torch.chunk(rms_result, len(group_ids), dim=-1)
        for id, chunk in zip(group_ids, per_shard_chunks):
            output_shards[id] = chunk.clone()

    return GoldenMapTensor(
        {i: t for i, t in enumerate(output_shards)}, input.mesh_shape
    )


def ttir_layer_norm_golden(
    input: GoldenMapTensor,
    weight: Optional[GoldenMapTensor],
    bias: Optional[GoldenMapTensor],
    normalized_shape: ArrayAttr,
    epsilon: FloatAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    normalized_shape = unpack_mlir_attr(normalized_shape)
    epsilon = unpack_mlir_attr(epsilon)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    input_float = input.float()

    return torch.nn.functional.layer_norm(
        input_float,
        normalized_shape=normalized_shape,
        weight=weight,
        bias=bias,
        eps=epsilon,
    ).to(output_dtype)


def typecast_golden(input_tensor: GoldenMapTensor, dtype) -> GoldenMapTensor:
    """
    Custom golden function for typecasting.

    Parameters
    ----------
    input_tensor : GoldenMapTensor
        Input tensor to typecast
    dtype : torch.dtype
        Target data type for typecasting

    Returns
    -------
    GoldenMapTensor
        Typecasted tensor
    """
    return input_tensor.to(dtype)


def argmax_golden(
    input_tensor: GoldenMapTensor, dim_arg=None, keep_dim=False
) -> GoldenMapTensor:
    """
    Custom golden function for argmax.

    Parameters
    ----------
    input_tensor : GoldenMapTensor
        Input tensor to find argmax of
    dim_arg : List[int], optional
        List of dimensions to reduce over. If None, reduces over all dimensions (default: None)
    keep_dim : bool, optional
        Whether to keep the reduced dimension (default: False)

    Returns
    -------
    GoldenMapTensor
        Indices of maximum values along specified dimension(s) as int32 tensor
    """
    if dim_arg is None:
        # Reduce over all dimensions - return flattened index
        result = torch.argmax(input_tensor, keepdim=keep_dim)
    elif len(dim_arg) == 1:
        # Single dimension reduction
        result = torch.argmax(input_tensor, dim=dim_arg[0], keepdim=keep_dim)
    else:
        # Multiple dimension reduction
        all_dims = list(range(input_tensor.dim()))

        # Keep reduction dimensions as given
        reduce_dims = dim_arg

        # Permute: move reduction dims to the end
        non_reduce_dims = [d for d in all_dims if d not in reduce_dims]
        perm_order = non_reduce_dims + reduce_dims
        permuted = input_tensor.permute(*perm_order)

        # Flatten reduction dimensions
        reduce_size = 1
        for d in reduce_dims:
            reduce_size *= input_tensor.size(d)

        # Reshape and apply argmax
        non_reduce_shape = [input_tensor.size(d) for d in non_reduce_dims]
        reshaped = permuted.reshape(*non_reduce_shape, reduce_size)
        result_flat = torch.argmax(reshaped, dim=-1)

        # Handle keepdim
        if keep_dim:
            output_shape = []
            for i in range(input_tensor.dim()):
                if i in reduce_dims:
                    output_shape.append(1)
                else:
                    output_shape.append(input_tensor.size(i))
            result = result_flat.reshape(*output_shape)
        else:
            result = result_flat

    return result.to(torch.int32)


def matmul_golden(
    a: GoldenMapTensor,
    b: GoldenMapTensor,
    transpose_a=False,
    transpose_b=False,
) -> GoldenMapTensor:
    """
    Custom golden function for matrix multiplication.

    Parameters
    ----------
    a : GoldenMapTensor
        First input tensor
    b : GoldenMapTensor
        Second input tensor
    transpose_a : bool, optional
        Whether to transpose tensor a (default: False)
    transpose_b : bool, optional
        Whether to transpose tensor b (default: False)

    Returns
    -------
    GoldenMapTensor
        Result of matrix multiplication
    """
    a = torch.transpose(a, -2, -1) if transpose_a else a
    b = torch.transpose(b, -2, -1) if transpose_b else b
    return torch.matmul(a, b)


def linear_golden(
    a: GoldenMapTensor,
    b: GoldenMapTensor,
    bias=None,
    transpose_a=False,
    transpose_b=False,
) -> GoldenMapTensor:
    """
    Custom golden function for linear transformation.

    Parameters
    ----------
    a : GoldenMapTensor
        First input tensor
    b : GoldenMapTensor
        Second input tensor
    bias : GoldenMapTensor, optional
        Optional bias tensor (default: None)
    transpose_a : bool, optional
        Whether to transpose tensor a (default: False)
    transpose_b : bool, optional
        Whether to transpose tensor b (default: False)

    Returns
    -------
    GoldenMapTensor
        Result of linear transformation with optional bias
    """
    a = torch.transpose(a, -2, -1) if transpose_a else a
    b = torch.transpose(b, -2, -1) if transpose_b else b
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
    lhs: GoldenMapTensor,
    rhs: GoldenMapTensor,
    batch_dims_lhs,
    contract_dims_lhs,
    batch_dims_rhs,
    contract_dims_rhs,
) -> GoldenMapTensor:
    """
    Custom golden function for dot_general operation.

    Parameters
    ----------
    lhs : GoldenMapTensor
        Left-hand side tensor
    rhs : GoldenMapTensor
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
    GoldenMapTensor
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
    input_tensor: GoldenMapTensor, scale, zero_point, dtype
) -> GoldenMapTensor:
    """
    Custom golden function for quantize operation.

    Parameters
    ----------
    input_tensor : GoldenMapTensor
        Input tensor to quantize
    scale : float
        Scale factor for quantization
    zero_point : int
        Zero point for quantization
    dtype : torch.dtype
        Target quantized data type

    Returns
    -------
    GoldenMapTensor
        Quantized tensor as integer representation
    """
    return torch.quantize_per_tensor(input_tensor, scale, zero_point, dtype).int_repr()


def requantize_golden(
    input_tensor: GoldenMapTensor, scale, zero_point, dtype
) -> GoldenMapTensor:
    """
    Custom golden function for requantize operation.

    Parameters
    ----------
    input_tensor : GoldenMapTensor
        Input quantized tensor to requantize
    scale : float
        Scale factor for requantization
    zero_point : int
        Zero point for requantization
    dtype : torch.dtype
        Target quantized data type

    Returns
    -------
    GoldenMapTensor
        Requantized tensor
    """
    return torch.quantize_per_tensor(
        torch.dequantize(input_tensor), scale, zero_point, dtype
    )


def logical_not_golden(input_tensor: GoldenMapTensor, **kwargs) -> GoldenMapTensor:
    """
    Golden function for logical_not operation.

    Elementwise logical NOT.

    Parameters
    ----------
    input_tensor : GoldenMapTensor
        Input tensor to invert logically.
    **kwargs : dict
        Keyword arguments (unused for this operation).

    Returns
    -------
    GoldenMapTensor
        Tensor with logical NOT of input_tensor, cast back to input dtype.
    """
    # Compute bool result then cast to match input dtype
    result_bool = torch.logical_not(input_tensor)
    return result_bool.to(input_tensor.dtype)


def equal_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, **kwargs
) -> GoldenMapTensor:
    """
    Golden function for equal (eq) operation.
    Used by TTNN dialect.

    Elementwise equality comparison.

    Parameters
    ----------
    input_tensor : GoldenMapTensor
        Left-hand side tensor.
    other_tensor : GoldenMapTensor
        Right-hand side tensor.

    Returns
    -------
    GoldenMapTensor
        Tensor with the same dtype as input_tensor containing the equality results.
    """
    result_bool = torch.eq(input_tensor, other_tensor)
    return result_bool.to(input_tensor.dtype)


def not_equal_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, **kwargs
) -> GoldenMapTensor:
    """
    Golden function for not_equal (ne) operation.

    Elementwise inequality comparison.

    Parameters
    ----------
    input_tensor : GoldenMapTensor
        Left-hand side tensor.
    other_tensor : GoldenMapTensor
        Right-hand side tensor.

    Returns
    -------
    GoldenMapTensor
        Tensor with the same dtype as input_tensor containing the inequality results.
    """
    result_bool = torch.ne(input_tensor, other_tensor)
    return result_bool.to(input_tensor.dtype)


def greater_equal_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, **kwargs
) -> GoldenMapTensor:
    """
    Golden function for greater_equal (ge) operation.

    Elementwise greater-than-or-equal comparison.

    Parameters
    ----------
    input_tensor : GoldenMapTensor
        Left-hand side tensor.
    other_tensor : GoldenMapTensor
        Right-hand side tensor.

    Returns
    -------
    GoldenMapTensor
        Tensor with the same dtype as input_tensor containing the comparison results.
    """
    result_bool = torch.ge(input_tensor, other_tensor)
    return result_bool.to(input_tensor.dtype)


def greater_than_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, **kwargs
) -> GoldenMapTensor:
    """
    Golden function for greater_than (gt) operation.
    Used by TTNN dialect.

    Elementwise greater-than comparison.

    Parameters
    ----------
    input_tensor : GoldenMapTensor
        Left-hand side tensor.
    other_tensor : GoldenMapTensor
        Right-hand side tensor.

    Returns
    -------
    GoldenMapTensor
        Tensor with the same dtype as input_tensor containing the comparison results.
    """
    result_bool = torch.gt(input_tensor, other_tensor)
    return result_bool.to(input_tensor.dtype)


def less_equal_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, **kwargs
) -> GoldenMapTensor:
    """
    Golden function for less_equal (le) operation.

    Elementwise less-than-or-equal comparison.

    Parameters
    ----------
    input_tensor : GoldenMapTensor
        Left-hand side tensor.
    other_tensor : GoldenMapTensor
        Right-hand side tensor.

    Returns
    -------
    GoldenMapTensor
        Tensor with the same dtype as input_tensor containing the comparison results.
    """
    result_bool = torch.le(input_tensor, other_tensor)
    return result_bool.to(input_tensor.dtype)


def less_than_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, **kwargs
) -> GoldenMapTensor:
    """
    Golden function for less_than (lt) operation.

    Elementwise less-than comparison.

    Parameters
    ----------
    input_tensor : GoldenMapTensor
        Left-hand side tensor.
    other_tensor : GoldenMapTensor
        Right-hand side tensor.

    Returns
    -------
    GoldenMapTensor
        Tensor with the same dtype as input_tensor containing the comparison results.
    """
    result_bool = torch.lt(input_tensor, other_tensor)
    return result_bool.to(input_tensor.dtype)


def logical_or_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, **kwargs
) -> GoldenMapTensor:
    """
    Golden function for logical_or operation.

    Elementwise logical OR.

    Parameters
    ----------
    input_tensor : GoldenMapTensor
        Left-hand side tensor.
    other_tensor : GoldenMapTensor
        Right-hand side tensor.

    Returns
    -------
    GoldenMapTensor
        Tensor with the same dtype as input_tensor containing the logical OR results.
    """
    result_bool = torch.logical_or(input_tensor, other_tensor)
    return result_bool.to(input_tensor.dtype)


def logical_xor_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, **kwargs
) -> GoldenMapTensor:
    """
    Golden function for logical_xor operation.

    Elementwise logical XOR.

    Parameters
    ----------
    input_tensor : GoldenMapTensor
        Left-hand side tensor.
    other_tensor : GoldenMapTensor
        Right-hand side tensor.

    Returns
    -------
    GoldenMapTensor
        Tensor with the same dtype as input_tensor containing the logical XOR results.
    """
    result_bool = torch.logical_xor(input_tensor, other_tensor)
    return result_bool.to(input_tensor.dtype)


def logical_left_shift_golden(
    input_tensor: GoldenMapTensor, shift_tensor: GoldenMapTensor, **kwargs
) -> GoldenMapTensor:
    """
    Golden function for logical left shift operation.

    Parameters
    ----------
    input_tensor : GoldenMapTensor
        Input tensor to be shifted.
    shift_tensor : GoldenMapTensor
        Tensor containing the number of bits to shift.

    Returns
    -------
    GoldenMapTensor
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
    input_tensor: GoldenMapTensor, shift_tensor: GoldenMapTensor, **kwargs
) -> GoldenMapTensor:
    """
    Golden function for logical right shift operation.

    Parameters
    ----------
    input_tensor : GoldenMapTensor
        Input tensor to be shifted.
    shift_tensor : GoldenMapTensor
        Tensor containing the number of bits to shift.

    Returns
    -------
    GoldenMapTensor
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


def min_golden(
    input_tensor: GoldenMapTensor, dim_arg=None, keep_dim=True
) -> GoldenMapTensor:
    """
    Golden function for min operation.

    Parameters
    ----------
    input_tensor : GoldenMapTensor
        Input tensor
    dim_arg : List[int], optional
        List of dimensions to reduce over. If None, reduces over all dimensions (default: None)
    keep_dim : bool, optional
        If True, retains reduced dimensions with length 1 (default: True)

    Returns
    -------
    GoldenMapTensor
        Tensor with minimum values along specified dimension(s) or global minimum
    """
    if dim_arg is None:
        # For all dimensions reduction
        result = torch.min(input_tensor)
        if keep_dim:
            # Reshape to match expected output with all dims = 1
            output_shape = [1] * input_tensor.dim()
            return result.reshape(*output_shape)
        else:
            return result
    elif len(dim_arg) == 1:
        # Single dimension reduction
        values, indices = torch.min(input_tensor, dim=dim_arg[0], keepdim=keep_dim)
        return values
    else:
        # Multiple dimensions - reduce sequentially from highest to lowest
        # Sort in descending order to maintain correct dimension indices
        sorted_dims = sorted(dim_arg, reverse=True)
        result = input_tensor
        for dim in sorted_dims:
            result, _ = torch.min(result, dim=dim, keepdim=keep_dim)
        return result


def prod_golden(
    input_tensor: GoldenMapTensor, dim_arg=None, keep_dim=False
) -> GoldenMapTensor:
    """
    Custom golden function for prod operation with conditional logic.

    Parameters
    ----------
    input_tensor : GoldenMapTensor
        Input tensor to compute product of
    dim_arg : List[int], optional
        List of dimensions to reduce over. If None, reduces over all dimensions (default: None)
    keep_dim : bool, optional
        Whether to keep the reduced dimension (default: False)

    Returns
    -------
    GoldenMapTensor
        Product of tensor elements along specified dimension(s) or global product
    """
    if dim_arg is None:
        # For all dimensions reduction
        result = torch.prod(input_tensor)
        if keep_dim:
            # Reshape to match expected output with all dims = 1
            output_shape = [1] * input_tensor.dim()
            return result.reshape(*output_shape)
        else:
            return result
    elif len(dim_arg) == 1:
        # Single dimension reduction
        return torch.prod(input_tensor, dim=dim_arg[0], keepdim=keep_dim)
    else:
        # Multiple dimensions - reduce sequentially from highest to lowest
        # Sort in descending order to maintain correct dimension indices
        sorted_dims = sorted(dim_arg, reverse=True)
        result = input_tensor
        for dim in sorted_dims:
            result = torch.prod(result, dim=dim, keepdim=keep_dim)
        return result


def embedding_golden(
    indices_tensor: GoldenMapTensor, weight_tensor: GoldenMapTensor
) -> GoldenMapTensor:
    """
    Custom golden function for embedding operation.

    Parameters
    ----------
    indices_tensor : GoldenMapTensor
        Tensor containing indices to look up
    weight_tensor : GoldenMapTensor
        Weight tensor containing embedding vectors. Can be "effectively 2D"
        with leading singleton dimensions (e.g., shape (1, 1, vocab, embed)).

    Returns
    -------
    GoldenMapTensor
        Embedded vectors corresponding to input indices
    """
    # Handle "effectively 2D" weights with leading singleton dimensions.
    # Reshape to 2D for torch.nn.Embedding which requires exactly 2D weights.
    vocab_size = weight_tensor.size(-2)
    embed_dim = weight_tensor.size(-1)
    weight_2d = weight_tensor.reshape(vocab_size, embed_dim)
    embedding = torch.nn.Embedding.from_pretrained(weight_2d)
    golden_typecast = indices_tensor.to(torch.int32)
    golden_input = torch.clamp(golden_typecast, 0, (vocab_size - 1))
    return embedding(golden_input)


def select_golden(
    input_tensor: GoldenMapTensor, dim, begin, length, stride
) -> GoldenMapTensor:
    """
    Custom golden function for select operation.

    Parameters
    ----------
    input_tensor : GoldenMapTensor
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
    GoldenMapTensor
        Selected tensor slice
    """
    end = begin + length - 1
    index = torch.tensor([begin, end])
    return torch.index_select(input_tensor, dim=dim, index=index)


def index_golden(
    input_tensor: GoldenMapTensor, dim, begin, end, step
) -> GoldenMapTensor:
    """
    Custom golden function for index operation.

    Parameters
    ----------
    input_tensor : GoldenMapTensor
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
    GoldenMapTensor
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
    input_tensor: GoldenMapTensor,
    start_indices_tensor: GoldenMapTensor,
    **kwargs,
) -> GoldenMapTensor:
    """
    Golden function for gather operation with TTIR parameter names.

    Parameters
    ----------
    input_tensor : GoldenMapTensor
        Input tensor to gather from
    start_indices_tensor : GoldenMapTensor
        Tensor containing starting indices
    **kwargs : dict
        Keyword arguments including gather attributes as MLIR attributes

    Returns
    -------
    GoldenMapTensor
        Gathered tensor
    """

    # helpers
    def _isGoldenMapTensor(x):
        return isinstance(x, GoldenMapTensor)

    def _first_shard(x):
        return x.shard_at(0) if _isGoldenMapTensor(x) else x

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
    if _isGoldenMapTensor(idx):
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
    input_tensor: GoldenMapTensor, tilize=True, **kwargs
) -> GoldenMapTensor:
    """
    Custom golden function for tilize operation.

    Parameters
    ----------
    input_tensor : GoldenMapTensor
        Input tensor to tilize
    tilize : bool, optional
        Tilize parameter (ignored, for compatibility) (default: True)

    Returns
    -------
    GoldenMapTensor
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
    input_tensor: GoldenMapTensor, tilize=False, **kwargs
) -> GoldenMapTensor:
    """
    Custom golden function for untilize operation.

    Parameters
    ----------
    input_tensor : GoldenMapTensor
        Input tensor to untilize
    tilize : bool, optional
        Tilize parameter (ignored, for compatibility) (default: False)

    Returns
    -------
    GoldenMapTensor
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
    in0: GoldenMapTensor, in1: GoldenMapTensor, scale_factor, mode="nearest"
) -> GoldenMapTensor:
    """
    Custom golden function for upsample2d operation.

    Parameters
    ----------
    in0 : GoldenMapTensor
        Input tensor to upsample
    in1 : GoldenMapTensor
        Output tensor specification
    scale_factor : Union[int, List[int]]
        Scaling factor for upsampling
    mode : str, optional
        Upsampling mode (default: "nearest")

    Returns
    -------
    GoldenMapTensor
        Upsampled 2D tensor
    """
    transposed_golden = torch.transpose(in0, 1, 3)
    golden_output_shape = in1.shape[1:-1]
    output = torch.nn.functional.interpolate(
        transposed_golden, size=golden_output_shape, mode=mode
    )
    return torch.transpose(output, 1, 3)


def fill_cache_golden(
    cache_tensor: GoldenMapTensor, input_tensor: GoldenMapTensor, **kwargs
) -> GoldenMapTensor:
    """
    Custom golden function for fill_cache operation.

    Parameters
    ----------
    cache_tensor : GoldenMapTensor
        Cache tensor to fill
    input_tensor : GoldenMapTensor
        Input tensor data
    **kwargs : dict
        Additional keyword arguments (batch_offset is ignored)

    Returns
    -------
    GoldenMapTensor
        Filled cache tensor
    """
    result = cache_tensor.clone()

    for device_id, shard in result.shard_map.items():
        shard[:, :, : input_tensor.shape[2], :] = input_tensor.shard_at(device_id)
    return result


def update_cache_golden(
    cache_tensor: GoldenMapTensor,
    update_tensor: GoldenMapTensor,
    indices_tensor,
    **kwargs,
) -> GoldenMapTensor:
    """
    Custom golden function for update_cache operation.

    Parameters
    ----------
    cache_tensor : GoldenMapTensor
        Cache tensor to update
    update_tensor : GoldenMapTensor
        Tensor containing update data
    indices_tensor : GoldenMapTensor
        Tensor containing update indices
    **kwargs : dict
        Additional keyword arguments (batch_offset is ignored)

    Returns
    -------
    GoldenMapTensor
        Updated cache tensor
    """
    result = cache_tensor.clone()

    for device_id, shard in result.shard_map.items():
        shard[:, :, : update_tensor.shape[2], :] = update_tensor.shard_at(device_id)
    return result


def get_dimension_size_golden(
    input_tensor: GoldenMapTensor, **kwargs
) -> GoldenMapTensor:
    """
    Golden function for get_dimension_size operation.

    Parameters
    ----------
    input_tensor : GoldenMapTensor
        Input tensor to get dimension size from
    **kwargs : dict
        Keyword arguments including 'dimension'

    Returns
    -------
    GoldenMapTensor
        Tensor containing the size of the specified dimension as int32
    """
    dimension = kwargs.get("dimension", 0)
    output_tensor = input_tensor.clone()

    for device_id, shard in input_tensor.shard_map.items():
        shard = torch.tensor(
            [input_tensor.shard_at(device_id).size(dimension)], dtype=torch.int32
        )
        output_tensor.shard_map[device_id] = shard

    return output_tensor


def mean_golden(input_tensor: GoldenMapTensor, **kwargs) -> GoldenMapTensor:
    """
    Golden function for mean operation with TTIR parameter names.

    Parameters
    ----------
    input_tensor : GoldenMapTensor
        Input tensor to compute mean of
    **kwargs : dict
        Keyword arguments including 'dim_arg' and 'keep_dim'

    Returns
    -------
    GoldenMapTensor
        Mean tensor
    """
    dim_arg = kwargs.get("dim_arg", [0])
    keep_dim = kwargs.get("keep_dim", True)
    return torch.mean(input_tensor, dim=dim_arg, keepdim=keep_dim)


def reduce_and_golden(input_tensor: GoldenMapTensor, **kwargs) -> GoldenMapTensor:
    """
    Golden function for reduce_and operation with TTIR parameter names.

    Parameters
    ----------
    input_tensor : GoldenMapTensor
        Input tensor to reduce
    **kwargs : dict
        Keyword arguments including 'dim_arg' and 'keep_dim'

    Returns
    -------
    GoldenMapTensor
        Reduced tensor
    """
    dim_arg = kwargs.get("dim_arg", [0])
    keep_dim = kwargs.get("keep_dim", True)
    return torch.all(input_tensor, dim=tuple(dim_arg), keepdim=keep_dim)


def reduce_or_golden(input_tensor: GoldenMapTensor, **kwargs) -> GoldenMapTensor:
    """
    Golden function for reduce_or operation with TTIR parameter names.

    Parameters
    ----------
    input_tensor : GoldenMapTensor
        Input tensor to reduce
    **kwargs : dict
        Keyword arguments including 'dim_arg' and 'keep_dim'

    Returns
    -------
    GoldenMapTensor
        Reduced tensor
    """
    dim_arg = kwargs.get("dim_arg", [0])
    keep_dim = kwargs.get("keep_dim", True)
    return torch.any(input_tensor, dim=tuple(dim_arg), keepdim=keep_dim).to(
        torch.float32
    )


def transpose_golden(input_tensor: GoldenMapTensor, **kwargs) -> GoldenMapTensor:
    """
    Golden function for transpose operation with TTIR parameter names.

    Parameters
    ----------
    input_tensor : GoldenMapTensor
        Input tensor
    **kwargs : dict
        Keyword arguments including 'dim0' and 'dim1'

    Returns
    -------
    GoldenMapTensor
        Transposed tensor
    """
    dim0 = kwargs.get("dim0", 0)
    dim1 = kwargs.get("dim1", 1)
    return torch.transpose(input_tensor, dim0, dim1)


def reshape_golden(input_tensor: GoldenMapTensor, **kwargs) -> GoldenMapTensor:
    """
    Golden function for reshape operation (TTIR/StableHLO).

    Supports static reshapes only. The target shape is resolved from the provided
    ``shape`` keyword argument (preferred). If unavailable, the function attempts
    to infer it from ``result_type`` or from an ``op`` handle present in ``kwargs``.

    Parameters
    ----------
    input_tensor : GoldenMapTensor
        Input tensor
    **kwargs : dict
        Keyword arguments including ``shape`` (preferred), or ``result_type`` / ``op`` for fallback

    Returns
    -------
    GoldenMapTensor
        Reshaped tensor
    """

    def _dim_to_int(dimension: Any) -> int:
        if isinstance(dimension, int):
            return dimension
        if hasattr(dimension, "value"):
            return int(dimension.value)
        return int(dimension)

    def _maybe_extract_shape_from_type(result_type: Any) -> Optional[Tuple[int, ...]]:
        if result_type is None or not hasattr(result_type, "shape"):
            return None
        return tuple(_dim_to_int(dim) for dim in result_type.shape)

    shape = kwargs.get("shape")
    if shape is None:
        shape = _maybe_extract_shape_from_type(kwargs.get("result_type"))

    if shape is None:
        op = kwargs.get("op")
        if op is not None:
            # Try OpView interface first.
            result = getattr(op, "result", None)
            if result is not None and hasattr(result, "type"):
                shape = _maybe_extract_shape_from_type(result.type)
            # Fall back to Operation-style results.
            if shape is None:
                results = getattr(op, "results", None)
                if results:
                    first_result = results[0]
                    result_type = getattr(first_result, "type", None)
                    shape = _maybe_extract_shape_from_type(result_type)

    if shape is None:
        # Backward-compatibility: if no shape/context is provided (as in Chisel CLI path),
        # treat it as identity reshape (use the input tensor's current shape).
        shape = input_tensor.shape

    shape_tuple = tuple(_dim_to_int(dim) for dim in shape)

    if any(dim == -1 for dim in shape_tuple):
        raise ValueError(
            "reshape_golden only supports static reshape (no -1 dimensions)."
        )

    return torch.reshape(input_tensor, shape_tuple)


def squeeze_golden(input_tensor: GoldenMapTensor, **kwargs) -> GoldenMapTensor:
    """
    Golden function for squeeze operation with TTIR parameter names.

    Parameters
    ----------
    input_tensor : GoldenMapTensor
        Input tensor
    **kwargs : dict
        Keyword arguments including 'dim'

    Returns
    -------
    GoldenMapTensor
        Squeezed tensor
    """
    dim = kwargs.get("dim", None)
    return torch.squeeze(input_tensor, dim=dim)


def unsqueeze_golden(input_tensor: GoldenMapTensor, **kwargs) -> GoldenMapTensor:
    """
    Golden function for unsqueeze operation with TTIR parameter names.

    Parameters
    ----------
    input_tensor : GoldenMapTensor
        Input tensor
    **kwargs : dict
        Keyword arguments including 'dim'

    Returns
    -------
    GoldenMapTensor
        Unsqueezed tensor
    """
    dim = kwargs.get("dim", 0)
    return torch.unsqueeze(input_tensor, dim=dim)


def clamp_scalar_golden(input_tensor: GoldenMapTensor, **kwargs) -> GoldenMapTensor:
    """
    Golden function for clamp_scalar operation with TTIR parameter names.

    Parameters
    ----------
    input_tensor : GoldenMapTensor
        Input tensor
    **kwargs : dict
        Keyword arguments including 'min' and 'max'

    Returns
    -------
    GoldenMapTensor
        Clamped tensor
    """
    min_val = kwargs.get("min", None)
    max_val = kwargs.get("max", None)
    return torch.clamp(input_tensor, min=min_val, max=max_val)


def clamp_tensor_golden(
    input_tensor: GoldenMapTensor,
    min_tensor: GoldenMapTensor,
    max_tensor: GoldenMapTensor,
    **kwargs,
) -> GoldenMapTensor:
    """
    Golden function for clamp_tensor operation with TTIR parameter names.

    Parameters
    ----------
    input_tensor : GoldenMapTensor
        Input tensor
    min_tensor : GoldenMapTensor
        Tensor specifying minimum values
    max_tensor : GoldenMapTensor
        Tensor specifying maximum values
    **kwargs : dict
        Additional keyword arguments

    Returns
    -------
    GoldenMapTensor
        Clamped tensor
    """
    return torch.min(torch.max(input_tensor, min_tensor), max_tensor)


def permute_golden(input_tensor: GoldenMapTensor, **kwargs) -> GoldenMapTensor:
    """
    Golden function for permute operation with TTIR parameter names.

    Parameters
    ----------
    input_tensor : GoldenMapTensor
        Input tensor
    **kwargs : dict
        Keyword arguments including 'permutation' as MLIR attribute

    Returns
    -------
    GoldenMapTensor
        Permuted tensor
    """

    permutation = kwargs.get("permutation", None)
    if permutation is None:
        return input_tensor

    permutation = unpack_mlir_attr(permutation)
    return torch.permute(input_tensor, tuple(permutation))


def leaky_relu_golden(input_tensor: GoldenMapTensor, **kwargs) -> GoldenMapTensor:
    """
    Golden function for leaky_relu operation with TTIR parameter names.

    Parameters
    ----------
    input_tensor : GoldenMapTensor
        Input tensor
    **kwargs : dict
        Keyword arguments including 'parameter'

    Returns
    -------
    GoldenMapTensor
        Leaky ReLU output
    """
    parameter = kwargs.get("parameter", 0.01)
    return torch.nn.functional.leaky_relu(input_tensor, negative_slope=parameter)


def silu_golden(input_tensor: GoldenMapTensor, **kwargs) -> GoldenMapTensor:
    """
    Golden function for silu operation with TTIR parameter names.

    Parameters
    ----------
    input_tensor : GoldenMapTensor
        Input tensor
    **kwargs : dict
        Additional keyword arguments

    Returns
    -------
    GoldenMapTensor
        SiLU output
    """
    return torch.nn.functional.silu(input_tensor)


def softmax_golden(input_tensor: GoldenMapTensor, **kwargs) -> GoldenMapTensor:
    """
    Golden function for silu operation with TTIR parameter names.

    Parameters
    ----------
    input_tensor : GoldenMapTensor
        Input tensor
    **kwargs : dict
        Additional keyword arguments

    Returns
    -------
    GoldenMapTensor
        Softmax output
    """
    dimension = kwargs.get("dim", 1)
    return torch.nn.functional.softmax(input_tensor, dim=dimension)


def index_golden(input_tensor: GoldenMapTensor, **kwargs) -> GoldenMapTensor:
    """
    Golden function for index operation with TTIR parameter names.

    Parameters
    ----------
    input_tensor : GoldenMapTensor
        Input tensor
    **kwargs : dict
        Keyword arguments including 'dim', 'begin', 'end', 'step'

    Returns
    -------
    GoldenMapTensor
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


def dynamic_slice_golden(
    input_tensor: GoldenMapTensor,
    **kwargs,
) -> GoldenMapTensor:
    """
    Golden function for dynamic_slice operation.

    Parameters
    ----------
    input_tensor : GoldenMapTensor
        Input tensor to slice
    *start_indices_tensors : GoldenMapTensor
        One tensor per dimension, each providing the start index for that dimension.
        Scalars or rank-1 tensors are supported; values are interpreted as integers.
    **kwargs : dict
        Keyword arguments including 'slice_sizes' as MLIR attribute

    Returns
    -------
    GoldenMapTensor
        Dynamically sliced tensor
    """

    start_indices_tensors = unpack_mlir_attr(kwargs.get("start_indices", []))
    slice_sizes = unpack_mlir_attr(kwargs.get("slice_sizes", []))
    rank = len(slice_sizes)
    assert rank == len(start_indices_tensors), "start_indices_tensors must match rank"

    # Extract start indices (use shard-0 for validation)
    starts: List[int] = []
    x0 = input_tensor.shard_at(0)
    for d, st in enumerate(start_indices_tensors):
        val0 = st if not isinstance(st, GoldenMapTensor) else st.shard_at(0)
        starts.append(val0)

        # Bounds check
        max_valid = x0.size(d) - slice_sizes[d]
        if starts[d] < 0 or starts[d] > max_valid:
            raise IndexError(
                f"dynamic_slice start index out of bounds for dim {d}: {starts[d]} not in [0,{max_valid}]"
            )

    # Build slices
    slicers = tuple(slice(starts[d], starts[d] + slice_sizes[d]) for d in range(rank))

    shard_map = {}
    for device_id, shard in input_tensor.shard_map.items():
        shard_map[device_id] = shard[slicers]

    return GoldenMapTensor(shard_map, input_tensor.mesh_shape)


def repeat_interleave_golden(
    input_tensor: GoldenMapTensor, **kwargs
) -> GoldenMapTensor:
    """
    Golden function for repeat_interleave operation with TTIR parameter names.

    Parameters
    ----------
    input_tensor : GoldenMapTensor
        Input tensor
    **kwargs : dict
        Keyword arguments including 'repeats' and 'dim'

    Returns
    -------
    GoldenMapTensor
        Repeated tensor
    """
    repeats = kwargs.get("repeats", 1)
    dim = kwargs.get("dim", 0)
    return torch.repeat_interleave(input_tensor, repeats, dim=dim)


def stablehlo_or_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, **kwargs
) -> GoldenMapTensor:
    """
    Golden function for StableHLO or operation.

    Supports both logical OR (for boolean tensors) and bitwise OR (for integer tensors).

    Parameters
    ----------
    input_tensor : GoldenMapTensor
        Left-hand side tensor.
    other_tensor : GoldenMapTensor
        Right-hand side tensor.

    Returns
    -------
    GoldenMapTensor
        Tensor containing the OR results.
    """
    if input_tensor.dtype == torch.bool:
        result_bool = torch.logical_or(input_tensor, other_tensor)
        return result_bool.to(input_tensor.dtype)
    else:
        return torch.bitwise_or(input_tensor, other_tensor)


def stablehlo_xor_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, **kwargs
) -> GoldenMapTensor:
    """
    Golden function for StableHLO xor operation.

    Supports both logical XOR (for boolean tensors) and bitwise XOR (for integer tensors).

    Parameters
    ----------
    input_tensor : GoldenMapTensor
        Left-hand side tensor.
    other_tensor : GoldenMapTensor
        Right-hand side tensor.

    Returns
    -------
    GoldenMapTensor
        Tensor containing the XOR results.
    """
    if input_tensor.dtype == torch.bool:
        result_bool = torch.logical_xor(input_tensor, other_tensor)
        return result_bool.to(input_tensor.dtype)
    else:
        return torch.bitwise_xor(input_tensor, other_tensor)


def stablehlo_not_golden(input_tensor: GoldenMapTensor, **kwargs) -> GoldenMapTensor:
    """
    Golden function for StableHLO not operation.

    Supports both logical NOT (for boolean tensors) and bitwise NOT (for integer tensors).

    Parameters
    ----------
    input_tensor : GoldenMapTensor
        Input tensor to invert.
    **kwargs : dict
        Keyword arguments (unused for this operation).

    Returns
    -------
    GoldenMapTensor
        Tensor containing the NOT of input_tensor.
    """
    if input_tensor.dtype == torch.bool:
        result_bool = torch.logical_not(input_tensor)
        return result_bool.to(input_tensor.dtype)
    else:
        return torch.bitwise_not(input_tensor)


################ Golden Utilities ###############


def apply_sharding(
    tensor: GoldenMapTensor,
    mesh_shape: Tuple[int],
    shard_dims: Tuple[Union[int, None]],
) -> GoldenMapTensor:
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
    return GoldenMapTensor(shard_dictionary, mesh_shape)


def apply_unsharding(
    tensor: GoldenMapTensor,
    mesh_shape: Tuple[int],
    shard_dims: Tuple[Union[int, None]],
) -> GoldenMapTensor:
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

    return GoldenMapTensor({0: shards[0]}, mesh_shape)


################ TTIR Op Golden Functions ###############


def ttir_rearrange_golden(
    input_tensor: GoldenMapTensor, pattern: StringAttr, output_type_mlir: Type
) -> GoldenMapTensor:
    pattern = unpack_mlir_attr(pattern)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    torch_fn = lambda t: torch.tensor(einops.rearrange(t.numpy(), pattern))
    result = GoldenMapTensor.__torch_function__(
        torch_fn, (GoldenMapTensor,), args=(input_tensor,)
    )
    return result.to(output_dtype)


def ttir_reduce_and_golden(
    input_tensor: GoldenMapTensor,
    dim_arg: ArrayAttr,
    keep_dim: BoolAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    dim_arg = unpack_mlir_attr(dim_arg)
    keep_dim = unpack_mlir_attr(keep_dim)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.all(input_tensor, dim=tuple(dim_arg), keepdim=keep_dim).to(
        output_dtype
    )


def ttir_repeat_golden(
    input: GoldenMapTensor,
    repeat_dimensions_attr: DenseI64ArrayAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    repeat_dimensions = unpack_mlir_attr(repeat_dimensions_attr)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return input.repeat(repeats=repeat_dimensions).to(output_dtype)


def ttir_arange_golden(
    shape: ArrayAttr,
    start: IntegerAttr,
    end: IntegerAttr,
    step: IntegerAttr,
    arange_dimension: IntegerAttr,
    mesh_shape_attr: ArrayAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    shape = unpack_mlir_attr(shape)
    start = unpack_mlir_attr(start)
    end = unpack_mlir_attr(end)
    step = unpack_mlir_attr(step)
    arange_dimension = unpack_mlir_attr(arange_dimension)
    mesh_shape = unpack_mlir_attr(mesh_shape_attr)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    result = torch.arange(start=start, end=end, step=step, dtype=torch.float32).to(
        output_dtype
    )

    broadcast_shape = [1] * len(shape)
    broadcast_shape[arange_dimension] = shape[arange_dimension]
    result = result.reshape(broadcast_shape)

    result = result.expand(shape).clone()

    return GoldenMapTensor(
        {i: result.clone() for i in range(mesh_shape[0] * mesh_shape[1])}, mesh_shape
    )


def ttir_cumsum_golden(
    input_tensor: GoldenMapTensor, dim: IntegerAttr, output_type_mlir: Type
) -> GoldenMapTensor:
    dim = unpack_mlir_attr(dim)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.cumsum(input_tensor, dim=dim).to(output_dtype)


def ttir_gather_golden(
    input_tensor: GoldenMapTensor,
    start_indices_tensor: GoldenMapTensor,
    offset_dims: DenseI64ArrayAttr,
    collapsed_slice_dims: DenseI64ArrayAttr,
    operand_batching_dims: DenseI64ArrayAttr,
    start_indices_batching_dims: DenseI64ArrayAttr,
    start_index_map: DenseI64ArrayAttr,
    index_vector_dim: IntegerAttr,
    slice_sizes: DenseI64ArrayAttr,
    indices_are_sorted: BoolAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:

    # helpers
    def _isGoldenMapTensor(x):
        return isinstance(x, GoldenMapTensor)

    def _first_shard(x):
        return x.shard_at(0) if _isGoldenMapTensor(x) else x

    def _assert_replicated(t):
        ref = t.shard_at(0)
        for device_id, shard in t.shard_map.items():
            if not torch.equal(shard, ref):
                raise ValueError("gather golden expects replicated tensors")

    # ----- unpack attrs -----
    offset_dims = unpack_mlir_attr(offset_dims)
    collapsed_slice_dims = unpack_mlir_attr(collapsed_slice_dims)
    operand_batching_dims = unpack_mlir_attr(operand_batching_dims)
    start_indices_batching_dims = unpack_mlir_attr(start_indices_batching_dims)
    start_index_map = unpack_mlir_attr(start_index_map)
    index_vector_dim = unpack_mlir_attr(index_vector_dim)
    slice_sizes = unpack_mlir_attr(slice_sizes)
    indices_are_sorted = unpack_mlir_attr(indices_are_sorted)

    x = input_tensor
    idx = start_indices_tensor
    device = x.device if hasattr(x, "device") else None

    # validate/comute using shard-0 (torch), then slice the wrapper
    if _isGoldenMapTensor(idx):
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
    elif index_vector_dim == idx0.ndim:
        # No explicit index vector dimension - each scalar is a single index
        batch_shape = idx0.shape
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

    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return gathered.to(output_dtype).to(device=device)


def ttir_ones_golden(
    shape: ArrayAttr, mesh_shape_attr: ArrayAttr, output_type_mlir: Type
) -> GoldenMapTensor:
    size = unpack_mlir_attr(shape)
    mesh_shape = unpack_mlir_attr(mesh_shape_attr)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    result = torch.ones(size, dtype=output_dtype)
    return GoldenMapTensor(
        {i: result.clone() for i in range(mesh_shape[0] * mesh_shape[1])}, mesh_shape
    )


def ttir_zeros_golden(
    shape: ArrayAttr, mesh_shape_attr: ArrayAttr, output_type_mlir: Type
) -> GoldenMapTensor:
    size = unpack_mlir_attr(shape)
    mesh_shape = unpack_mlir_attr(mesh_shape_attr)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    result = torch.zeros(size, dtype=output_dtype)
    return GoldenMapTensor(
        {i: result.clone() for i in range(mesh_shape[0] * mesh_shape[1])}, mesh_shape
    )


def ttir_rand_golden(
    size: ArrayAttr,
    low: FloatAttr,
    high: FloatAttr,
    seed: IntegerAttr,
    mesh_shape_attr: ArrayAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    size = unpack_mlir_attr(size)
    low = unpack_mlir_attr(low)
    high = unpack_mlir_attr(high)
    seed = unpack_mlir_attr(seed)
    mesh_shape = unpack_mlir_attr(mesh_shape_attr)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)

    gen = torch.Generator()
    gen.manual_seed(seed)
    base = torch.rand(size, generator=gen, dtype=torch.bfloat16)
    rand_tensor = (base * (high - low) + low).to(output_dtype)
    return GoldenMapTensor(
        {i: rand_tensor.clone() for i in range(mesh_shape[0] * mesh_shape[1])},
        mesh_shape,
    )


def ttir_dropout_golden(
    input_tensor: GoldenMapTensor,
    prob: FloatAttr,
    scale: FloatAttr,
    seed: IntegerAttr,
    use_per_device_seed: BoolAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    prob_val = unpack_mlir_attr(prob)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)

    return torch.dropout(input_tensor, prob_val, True).to(output_dtype)


def ttir_cos_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.cos(input_tensor).to(output_dtype)


def ttir_sin_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.sin(input_tensor).to(output_dtype)


def ttir_sqrt_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.sqrt(input_tensor).to(output_dtype)


def ttir_pow_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.pow(input_tensor, other_tensor).to(output_dtype)


def ttir_ge_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.ge(input_tensor, other_tensor).to(output_dtype)


def ttir_lt_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.lt(input_tensor, other_tensor).to(output_dtype)


def ttir_le_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.le(input_tensor, other_tensor).to(output_dtype)


def ttir_bitwise_and_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.bitwise_and(input_tensor, other_tensor).to(output_dtype)


def ttir_minimum_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.minimum(input_tensor, other_tensor).to(output_dtype)


def ttir_logical_and_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.logical_and(input_tensor, other_tensor).to(output_dtype)


def ttir_logical_right_shift_golden(
    input_tensor: GoldenMapTensor, shift_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    input_int64 = input_tensor.to(torch.int64)
    shift_int64 = shift_tensor.to(torch.int64)
    input_unsigned = torch.bitwise_and(input_int64, 0xFFFFFFFF)
    result = torch.bitwise_right_shift(input_unsigned, shift_int64)
    return torch.bitwise_and(result, 0xFFFFFFFF).to(output_dtype)


def ttir_slice_golden(
    input_tensor: GoldenMapTensor,
    begins: ArrayAttr,
    ends: ArrayAttr,
    step: ArrayAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    # Unpack MLIR attributes
    begins = unpack_mlir_attr(begins)
    ends = unpack_mlir_attr(ends)
    step = unpack_mlir_attr(step)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)

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

    return GoldenMapTensor(shard_map, input_tensor.mesh_shape).to(output_dtype)


def ttir_div_golden(
    lhs: GoldenMapTensor, rhs: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.div(lhs, rhs).to(lhs.dtype).to(output_dtype)


def ttir_sum_golden(
    input_tensor: GoldenMapTensor,
    dim_arg_attr: ArrayAttr,
    keep_dim_attr: BoolAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    dim_arg = unpack_mlir_attr(dim_arg_attr)
    keep_dim = unpack_mlir_attr(keep_dim_attr)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.sum(input_tensor, dim=dim_arg, keepdim=keep_dim).to(output_dtype)


def ttir_reshape_golden(
    input_tensor: GoldenMapTensor, shape_attr: ArrayAttr, output_type_mlir: Type
) -> GoldenMapTensor:
    new_shape = unpack_mlir_attr(shape_attr)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.reshape(input_tensor, new_shape).clone().to(output_dtype)


def ttir_broadcast_golden(
    input_tensor: GoldenMapTensor,
    broadcast_dimensions_attr: DenseI64ArrayAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    broadcast_dimensions = unpack_mlir_attr(broadcast_dimensions_attr)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    input_shape = input_tensor.shape

    shape = []
    for i in range(len(broadcast_dimensions)):
        if broadcast_dimensions[i] != 1:
            shape.append(broadcast_dimensions[i])
        else:
            shape.append(input_shape[i])

    return torch.broadcast_to(input_tensor, shape).to(output_dtype)


def ttir_permute_golden(
    input_tensor: GoldenMapTensor,
    permutation_attr: DenseI64ArrayAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    permutation = unpack_mlir_attr(permutation_attr)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.permute(input_tensor, permutation).to(output_dtype)


def ttir_dot_general_golden(
    lhs: GoldenMapTensor,
    rhs: GoldenMapTensor,
    batch_dims_lhs_attr: DenseI64ArrayAttr,
    contract_dims_lhs_attr: DenseI64ArrayAttr,
    batch_dims_rhs_attr: DenseI64ArrayAttr,
    contract_dims_rhs_attr: DenseI64ArrayAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    batch_dims_lhs = unpack_mlir_attr(batch_dims_lhs_attr)
    contract_dims_lhs = unpack_mlir_attr(contract_dims_lhs_attr)
    batch_dims_rhs = unpack_mlir_attr(batch_dims_rhs_attr)
    contract_dims_rhs = unpack_mlir_attr(contract_dims_rhs_attr)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)

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
    return result.to(output_dtype)


def ttir_pad_golden(
    input_tensor: GoldenMapTensor,
    padding: DenseI32ArrayAttr,
    value: FloatAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    padding = unpack_mlir_attr(padding)
    value = unpack_mlir_attr(value)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)

    golden_padding = []
    for i in range(len(padding) // 2):
        golden_padding.append(padding[-((2 * i) + 2)])
        golden_padding.append(padding[-((2 * i) + 1)])

    return torch.nn.functional.pad(
        input_tensor, pad=golden_padding, mode="constant", value=value
    ).to(output_dtype)


def ttir_constant_golden(
    value: DenseElementsAttr, mesh_shape_attr: ArrayAttr
) -> GoldenMapTensor:
    shape = list(value.type.shape)
    mesh_shape = unpack_mlir_attr(mesh_shape_attr)
    dtype = mlir_type_to_torch_dtype(value.type.element_type)

    if value.is_splat:
        value = value.get_splat_value()
        torch_tensor = torch.full(shape, value.value, dtype=dtype)
    else:
        flat_values = [elem for elem in value]
        torch_tensor = torch.tensor(flat_values, dtype=dtype).reshape(shape)

    result = torch_tensor.reshape(shape)
    return GoldenMapTensor(
        {i: result.clone() for i in range(mesh_shape[0] * mesh_shape[1])}, mesh_shape
    )


def ttir_convolution_golden(
    lhs: GoldenMapTensor,
    rhs: GoldenMapTensor,
    bias: Optional[GoldenMapTensor],
    window_strides_attr: DenseI64ArrayAttr,
    padding_attr: DenseI64ArrayAttr,
    input_dilation_attr: DenseI64ArrayAttr,
    weight_dilation_attr: DenseI64ArrayAttr,
    window_reversal_attr: DenseBoolArrayAttr,
    convolution_layout_attr: ConvolutionLayoutAttr,
    feature_group_count_attr: IntegerAttr,
    batch_group_count_attr: IntegerAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    input_tensor = lhs.clone()
    weight = rhs.clone()
    window_strides = unpack_mlir_attr(window_strides_attr)
    padding = unpack_mlir_attr(padding_attr)
    input_dilation = unpack_mlir_attr(input_dilation_attr)
    weight_dilation = unpack_mlir_attr(weight_dilation_attr)
    window_reversal = unpack_mlir_attr(window_reversal_attr)
    feature_group_count = unpack_mlir_attr(feature_group_count_attr)
    batch_group_count = unpack_mlir_attr(batch_group_count_attr)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    convolution_layout = ttir.ir.ConvolutionLayoutAttr.maybe_downcast(
        convolution_layout_attr
    )
    input_batch = convolution_layout.input_batch
    input_feature = convolution_layout.input_feature
    input_spatial_dimensions = convolution_layout.input_spatial_dimensions
    output_batch = convolution_layout.output_batch
    output_feature = convolution_layout.output_feature
    output_spatial_dimensions = convolution_layout.output_spatial_dimensions
    kernel_output_feature = convolution_layout.kernel_output_feature
    kernel_input_feature = convolution_layout.kernel_input_feature
    kernel_spatial_dimensions = convolution_layout.kernel_spatial_dimensions

    # Current layout is defined by the positions
    # We need to permute to NCHW: [batch, feature, spatial_0, spatial_1, ...]
    current_layout = [None] * input_tensor.ndim
    current_layout[input_batch] = 0  # batch goes to position 0
    current_layout[input_feature] = 1  # feature goes to position 1
    for i, spatial_dim in enumerate(input_spatial_dimensions):
        current_layout[spatial_dim] = 2 + i  # spatial dims go to positions 2, 3, ...

    # Check if we need to permute (i.e., if current_layout != [0, 1, 2, 3, ...])
    if current_layout != list(range(input_tensor.ndim)):
        # Create inverse permutation to go from current layout to NCHW
        permutation = [current_layout.index(i) for i in range(input_tensor.ndim)]
        input_tensor = input_tensor.permute(permutation)

    # Similarly for output, we need to know how to permute back
    # Output permutation: from NCHW back to output layout
    output_permutation = [None] * (len(output_spatial_dimensions) + 2)
    output_permutation[output_batch] = 0
    output_permutation[output_feature] = 1
    for i, spatial_dim in enumerate(output_spatial_dimensions):
        output_permutation[spatial_dim] = 2 + i

    # Handle weight/kernel layout transformation
    # PyTorch expects weights in [output_channels, input_channels, H, W] format
    # Create layout for weight tensor: [output_feat, input_feat, spatial_0, spatial_1, ...]
    weight_layout = [None] * weight.ndim
    weight_layout[kernel_output_feature] = 0  # output feature goes to position 0
    weight_layout[kernel_input_feature] = 1  # input feature goes to position 1
    for i, spatial_dim in enumerate(kernel_spatial_dimensions):
        weight_layout[spatial_dim] = 2 + i  # spatial dims go to positions 2, 3, ...

    # Check if we need to permute weight
    if weight_layout != list(range(weight.ndim)):
        weight_permutation = [weight_layout.index(i) for i in range(weight.ndim)]
        weight = weight.permute(weight_permutation)

    # Extract only spatial dimensions from strides and dilations
    # TTIR uses 4D strides/dilations [batch, channel, height, width]
    # PyTorch conv2d expects 2D [height, width]
    if len(window_strides) == 4:
        stride = [window_strides[2], window_strides[3]]  # Extract spatial dims
    elif len(window_strides) == 2:
        stride = window_strides
    else:
        stride = [1, 1]

    if len(weight_dilation) == 4:
        dilation = [weight_dilation[2], weight_dilation[3]]  # Extract spatial dims
    elif len(weight_dilation) == 2:
        dilation = weight_dilation
    else:
        dilation = [1, 1]

    # Convert padding from [top, left, bottom, right] to PyTorch format [height, width]
    # PyTorch expects symmetric padding, so we check if padding is symmetric
    if len(padding) == 4:
        top, left, bottom, right = padding
        if top == bottom and left == right:
            torch_padding = [top, left]
        else:
            # For asymmetric padding, we need to manually pad the input
            import torch.nn.functional as F

            # PyTorch F.pad expects padding in reverse order: [left, right, top, bottom]
            manual_padding = [left, right, top, bottom]
            input_tensor = F.pad(input_tensor, manual_padding, mode="constant", value=0)
            torch_padding = [0, 0]
    elif len(padding) == 2:
        torch_padding = padding
    else:
        torch_padding = [0, 0]

    # Handle bias
    if bias is not None:
        bias = bias.squeeze()

    # Now input_tensor is in NCHW format, call PyTorch conv2d directly
    groups = feature_group_count

    # Ensure input and weight have matching dtypes for conv2d
    # Use the input dtype as the common dtype since it comes from the computation chain
    if input_tensor.dtype != weight.dtype:
        weight = weight.to(input_tensor.dtype)
        if bias is not None:
            bias = bias.to(input_tensor.dtype)

    result = torch.nn.functional.conv2d(
        input_tensor,
        weight,
        bias=bias,
        stride=tuple(stride) if isinstance(stride, list) else stride,
        padding=(
            tuple(torch_padding) if isinstance(torch_padding, list) else torch_padding
        ),
        dilation=tuple(dilation) if isinstance(dilation, list) else dilation,
        groups=groups,
    )

    # Permute output back to the expected output layout if needed
    if output_permutation != list(range(result.ndim)):
        result = result.permute(output_permutation)

    return result.to(output_dtype)


def ttir_batch_norm_inference_golden(
    input_tensor: GoldenMapTensor,
    scale: GoldenMapTensor,
    offset: GoldenMapTensor,
    mean: GoldenMapTensor,
    variance: GoldenMapTensor,
    epsilon_attr: FloatAttr,
    dimension_attr: IntegerAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    epsilon = unpack_mlir_attr(epsilon_attr)
    dim = unpack_mlir_attr(dimension_attr)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    perm = list(range(input_tensor.ndim))
    perm[1], perm[dim] = perm[dim], perm[1]
    cloned_tensor = input_tensor.clone()
    permuted_tensor = cloned_tensor.permute(perm)
    result = torch.nn.functional.batch_norm(
        permuted_tensor,
        running_mean=mean,
        running_var=variance,
        weight=scale,
        bias=offset,
        training=False,
        eps=epsilon,
    )
    inv_perm = [perm.index(i) for i in range(len(perm))]
    result = result.permute(inv_perm)
    return result.to(output_dtype)


def ttir_batch_norm_training_golden(
    input_tensor: GoldenMapTensor,
    scale: GoldenMapTensor,
    offset: GoldenMapTensor,
    running_mean: GoldenMapTensor,
    running_variance: GoldenMapTensor,
    epsilon_attr: FloatAttr,
    dimension_attr: IntegerAttr,
    momentum_attr: FloatAttr,
    output_type_mlir: Type,
    mean_output_type_mlir: Type,
    variance_output_type_mlir: Type,
) -> Tuple[GoldenMapTensor, GoldenMapTensor, GoldenMapTensor]:
    epsilon = unpack_mlir_attr(epsilon_attr)
    dim = unpack_mlir_attr(dimension_attr)
    momentum = unpack_mlir_attr(momentum_attr)
    perm = list(range(input_tensor.ndim))
    perm[1], perm[dim] = perm[dim], perm[1]
    permuted_tensor = input_tensor.permute(perm)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    mean_output_type = mlir_type_to_torch_dtype(mean_output_type_mlir)
    variance_output_type = mlir_type_to_torch_dtype(variance_output_type_mlir)

    # Compute batch statistics
    # PyTorch batch_norm uses biased variance (unbiased=False) during training
    # Calculate mean and variance across all dimensions except the channel dimension (dim 1 after permute)
    # For NCHW format: reduce over dims [0, 2, 3] (batch, height, width), keep dim 1 (channels)
    if permuted_tensor.ndim == 4:
        # NCHW format - compute mean step by step to ensure it works with GoldenMapTensor
        # First reduce over spatial dimensions [2, 3] (H, W)
        spatial_mean = torch.mean(permuted_tensor, dim=[2, 3])  # [N, C]
        # Then reduce over batch dimension [0]
        batch_mean = torch.mean(spatial_mean, dim=0)  # [C]

        # For variance, use the same approach
        spatial_var_sum = torch.sum(
            torch.pow(
                torch.sub(permuted_tensor, torch.reshape(batch_mean, [1, -1, 1, 1])), 2
            ),
            dim=[0, 2, 3],
        )
        batch_var = torch.div(
            spatial_var_sum,
            permuted_tensor.shape[0]
            * permuted_tensor.shape[2]
            * permuted_tensor.shape[3],
        )
    else:
        # General case: reduce over all dims except channel dim (1)
        reduce_dims = [0] + list(range(2, permuted_tensor.ndim))
        batch_mean = torch.mean(permuted_tensor, dim=reduce_dims)
        batch_var = torch.var(permuted_tensor, dim=reduce_dims, unbiased=False)

    # Manually compute normalized output: (x - mean) / sqrt(var + eps) * scale + offset
    # Reshape mean and var for broadcasting using torch.reshape
    shape = [1, -1] + [1] * (permuted_tensor.ndim - 2)
    batch_mean_reshaped = torch.reshape(batch_mean, shape)
    batch_var_reshaped = torch.reshape(batch_var, shape)
    scale_reshaped = torch.reshape(scale, shape)
    offset_reshaped = torch.reshape(offset, shape)

    # Normalize using torch functions (GoldenMapTensor requires torch.* functions)
    centered = torch.sub(permuted_tensor, batch_mean_reshaped)
    std = torch.sqrt(torch.add(batch_var_reshaped, epsilon))
    normalized = torch.div(centered, std)
    scaled = torch.mul(normalized, scale_reshaped)
    result = torch.add(scaled, offset_reshaped)

    # Permute result back to original dimension order
    inv_perm = [perm.index(i) for i in range(len(perm))]
    result = result.permute(inv_perm)

    # Update running statistics using momentum
    # running_mean = momentum * batch_mean + (1 - momentum) * running_mean
    # running_variance = momentum * batch_variance + (1 - momentum) * running_variance
    updated_running_mean = torch.add(
        torch.mul(batch_mean, momentum), torch.mul(running_mean, 1 - momentum)
    )
    updated_running_var = torch.add(
        torch.mul(batch_var, momentum), torch.mul(running_variance, 1 - momentum)
    )

    return (
        result.to(output_dtype),
        updated_running_mean.to(mean_output_type),
        updated_running_var.to(variance_output_type),
    )


def ttir_ne_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    result_bool = torch.ne(input_tensor, other_tensor)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return result_bool.to(output_dtype)


def ttir_logical_not_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    result_bool = torch.logical_not(input_tensor)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return result_bool.to(output_dtype)


def ttir_split_query_key_value_and_split_heads_golden(
    input_tensor: GoldenMapTensor,
    kv_input_tensor: Optional[GoldenMapTensor],
    num_heads_attr: IntegerAttr,
    num_kv_heads_attr: Optional[IntegerAttr],
    transpose_key_attr: BoolAttr,
    query_output_type_mlir: Type,
    key_output_type_mlir: Type,
    value_output_type_mlir: Type,
) -> Tuple[GoldenMapTensor, GoldenMapTensor, GoldenMapTensor]:
    """
    Golden function for split_query_key_value_and_split_heads operation.

    This operation splits fused QKV tensors and reshapes them for attention computation.

    Parameters
    ----------
    input_tensor : GoldenMapTensor
        For MHA: [batch, seq, 3 * hidden_size] containing fused Q, K, V
        For GQA: [batch, seq, hidden_size] containing Q only
    kv_input_tensor : Optional[GoldenMapTensor]
        For GQA: [batch, seq, 2 * hidden_size] containing fused K, V
        For MHA: None
    num_heads_attr : IntegerAttr
        Number of query attention heads
    num_kv_heads_attr : Optional[IntegerAttr]
        Number of key/value heads (for GQA). If None, equals num_heads (MHA case)
    transpose_key_attr : BoolAttr
        Whether to transpose the key tensor
    query_output_type_mlir : Type
        Output type for query tensor
    key_output_type_mlir : Type
        Output type for key tensor
    value_output_type_mlir : Type
        Output type for value tensor

    Returns
    -------
    Tuple[GoldenMapTensor, GoldenMapTensor, GoldenMapTensor]
        query: [batch, num_heads, seq, head_size]
        key: [batch, num_kv_heads, seq, head_size] or [batch, num_kv_heads, head_size, seq] if transpose_key
        value: [batch, num_kv_heads, seq, head_size]
    """
    num_heads = unpack_mlir_attr(num_heads_attr)
    num_kv_heads = (
        unpack_mlir_attr(num_kv_heads_attr) if num_kv_heads_attr else num_heads
    )
    transpose_key = unpack_mlir_attr(transpose_key_attr)
    query_output_dtype = mlir_type_to_torch_dtype(query_output_type_mlir)
    key_output_dtype = mlir_type_to_torch_dtype(key_output_type_mlir)
    value_output_dtype = mlir_type_to_torch_dtype(value_output_type_mlir)

    batch_size = input_tensor.shape[0]
    seq_len = input_tensor.shape[1]

    if kv_input_tensor is None:
        # MHA case: input_tensor contains fused Q, K, V
        # Shape: [batch, seq, 3 * hidden_size]
        hidden_size = input_tensor.shape[2] // 3
        head_size = hidden_size // num_heads

        # Slice Q, K, V from fused tensor
        q = input_tensor[:, :, :hidden_size]
        k = input_tensor[:, :, hidden_size : 2 * hidden_size]
        v = input_tensor[:, :, 2 * hidden_size :]

        # Reshape: [batch, seq, hidden] -> [batch, seq, num_heads, head_size]
        q = q.reshape(batch_size, seq_len, num_heads, head_size)
        k = k.reshape(batch_size, seq_len, num_kv_heads, head_size)
        v = v.reshape(batch_size, seq_len, num_kv_heads, head_size)
    else:
        # GQA case: separate Q and KV tensors
        # input_tensor: [batch, seq, q_hidden_size]
        # kv_input_tensor: [batch, seq, 2 * kv_hidden_size]
        q_hidden_size = input_tensor.shape[2]
        head_size = q_hidden_size // num_heads
        kv_hidden_size = kv_input_tensor.shape[2] // 2

        # Q directly from input_tensor
        q = input_tensor.reshape(batch_size, seq_len, num_heads, head_size)

        # Slice K, V from kv_input_tensor
        k = kv_input_tensor[:, :, :kv_hidden_size]
        v = kv_input_tensor[:, :, kv_hidden_size:]

        # Reshape K, V: [batch, seq, kv_hidden] -> [batch, seq, num_kv_heads, head_size]
        k = k.reshape(batch_size, seq_len, num_kv_heads, head_size)
        v = v.reshape(batch_size, seq_len, num_kv_heads, head_size)

    # Permute: [batch, seq, num_heads, head_size] -> [batch, num_heads, seq, head_size]
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)

    # Optionally transpose key: [batch, num_heads, seq, head_size] -> [batch, num_heads, head_size, seq]
    if transpose_key:
        k = k.transpose(-2, -1)

    return (
        q.to(query_output_dtype),
        k.to(key_output_dtype),
        v.to(value_output_dtype),
    )


def ttir_max_golden(
    input_tensor: GoldenMapTensor,
    dim_arg_attr: ArrayAttr,
    keep_dim_attr: BoolAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    dim_arg = unpack_mlir_attr(dim_arg_attr)
    keep_dim = unpack_mlir_attr(keep_dim_attr)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)

    if dim_arg is None:
        # For all dimensions reduction
        result = torch.max(input_tensor)
        if keep_dim:
            # Reshape to match expected output with all dims = 1
            output_shape = [1] * input_tensor.dim()
            return result.reshape(*output_shape).to(output_dtype)
        else:
            return result.to(output_dtype)
    elif len(dim_arg) == 1:
        # Single dimension reduction
        values, indices = torch.max(input_tensor, dim=dim_arg[0], keepdim=keep_dim)
        return values.to(output_dtype)
    else:
        # Multiple dimensions - reduce sequentially from highest to lowest
        # Sort in descending order to maintain correct dimension indices
        sorted_dims = sorted(dim_arg, reverse=True)
        result = input_tensor
        for dim in sorted_dims:
            result, _ = torch.max(result, dim=dim, keepdim=keep_dim)
        return result.to(output_dtype)


def ttir_reduce_or_golden(
    input_tensor: GoldenMapTensor,
    dim_arg_attr: ArrayAttr,
    keep_dim_attr: BoolAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    dim_arg = unpack_mlir_attr(dim_arg_attr)
    keep_dim = unpack_mlir_attr(keep_dim_attr)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.any(input_tensor, dim=tuple(dim_arg), keepdim=keep_dim).to(
        output_dtype
    )


def ttir_clamp_tensor_golden(
    input_tensor: GoldenMapTensor,
    min_tensor: GoldenMapTensor,
    max_tensor: GoldenMapTensor,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.min(torch.max(input_tensor, min_tensor), max_tensor).to(output_dtype)


def ttir_full_golden(
    shape_attr: DenseI32ArrayAttr,
    fill_value_attr: Union[IntegerAttr, FloatAttr],
    mesh_shape_attr: ArrayAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    shape = unpack_mlir_attr(shape_attr)
    fill_value = unpack_mlir_attr(fill_value_attr)
    mesh_shape = unpack_mlir_attr(mesh_shape_attr)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    tensor = torch.full(shape, fill_value).to(output_dtype)
    return GoldenMapTensor(
        {i: tensor.clone() for i in range(mesh_shape[0] * mesh_shape[1])}, mesh_shape
    )


def ttir_concat_golden(
    input_tensors: List[GoldenMapTensor], dim_attr: IntegerAttr, output_type_mlir: Type
) -> GoldenMapTensor:
    dim = unpack_mlir_attr(dim_attr)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    if isinstance(input_tensors, tuple):
        return torch.concat(input_tensors, dim=dim).to(output_dtype)
    else:
        return torch.concat([input_tensors], dim=dim).to(output_dtype)


def ttir_max_pool2d_with_indices(
    input_tensor: GoldenMapTensor,
    kernel_attr: DenseI32ArrayAttr,
    stride_attr: DenseI32ArrayAttr,
    padding_attr: DenseI32ArrayAttr,
    dilation_attr: DenseI32ArrayAttr,
    ceil_mode_attr: BoolAttr,
    output_type_mlir: Type,
) -> Tuple[GoldenMapTensor, GoldenMapTensor]:
    kernel = unpack_mlir_attr(kernel_attr)
    stride = unpack_mlir_attr(stride_attr)
    padding = unpack_mlir_attr(padding_attr)
    dilation = unpack_mlir_attr(dilation_attr)
    ceil_mode = unpack_mlir_attr(ceil_mode_attr)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)

    # Assert that padding padding top+bottom and left+right are equal for both dimensions
    assert (
        padding[0] == padding[2]
    ), "Asymmetric padding not supported in height dimension"
    assert (
        padding[1] == padding[3]
    ), "Asymmetric padding not supported in width dimension"

    # TTMLIR uses NHWC format, but PyTorch expects NCHW format
    # Transpose input from NHWC to NCHW
    input_tensor_clone = input_tensor.clone()
    input_tensor_nchw = input_tensor_clone.transpose(-1, -2).transpose(-2, -3)

    output, indices = torch.nn.functional.max_pool2d(
        input_tensor_nchw,
        kernel_size=kernel,
        stride=stride,
        padding=((padding[0], padding[1])),
        dilation=dilation,
        ceil_mode=ceil_mode,
        return_indices=True,
    )

    # Transpose output back from NCHW to NHWC
    output_nhwc = output.transpose(-2, -3).transpose(-1, -2)
    indices_nhwc = indices.transpose(-2, -3).transpose(-1, -2)
    return output_nhwc.to(output_dtype), indices_nhwc.to(torch.int64)


def ttir_scatter_golden(
    input_tensor: GoldenMapTensor,
    index: GoldenMapTensor,
    source: GoldenMapTensor,
    dim: IntegerAttr,
    scatter_reduce_type_attr: ReduceTypeAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    dim_value = unpack_mlir_attr(dim)
    scatter_reduce_type = ttcore.ir.ReduceTypeAttr.maybe_downcast(
        scatter_reduce_type_attr
    ).value
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    index_copy = index.clone()
    index_copy = index_copy.to(torch.int64)

    if scatter_reduce_type == ttcore.ir.ReduceType.Sum:
        out_tensor = torch.scatter_reduce(
            input_tensor, dim_value, index_copy, source, reduce="sum"
        )
    elif scatter_reduce_type == ttcore.ir.ReduceType.Prod:
        out_tensor = torch.scatter_reduce(
            input_tensor, dim_value, index_copy, source, reduce="prod"
        )
    elif scatter_reduce_type == ttcore.ir.ReduceType.Max:
        out_tensor = torch.scatter_reduce(
            input_tensor, dim_value, index_copy, source, reduce="amax"
        )
    elif scatter_reduce_type == ttcore.ir.ReduceType.Min:
        out_tensor = torch.scatter_reduce(
            input_tensor, dim_value, index_copy, source, reduce="amin"
        )
    elif scatter_reduce_type == ttcore.ir.ReduceType.Invalid:
        out_tensor = torch.scatter(input_tensor, dim_value, index_copy, source)
    else:
        raise ValueError(f"Unsupported scatter reduce type: {scatter_reduce_type}")

    return out_tensor.to(output_dtype)


def ttir_reverse_golden(
    input_tensor: GoldenMapTensor,
    dimensions_attr: DenseI64ArrayAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    dimensions = unpack_mlir_attr(dimensions_attr)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.flip(input_tensor, dimensions).to(output_dtype)


def stablehlo_sort_golden(
    input_tensor: GoldenMapTensor,
    dimension_attr: IntegerAttr,
    is_stable_attr: BoolAttr,
    descending_attr: BoolAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    dimension = unpack_mlir_attr(dimension_attr)
    is_stable = unpack_mlir_attr(is_stable_attr)
    descending = unpack_mlir_attr(descending_attr)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)

    values, _ = torch.sort(
        input_tensor, dim=dimension, descending=descending, stable=is_stable
    )
    return values.to(output_dtype)


def ttir_equal_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    result_bool = torch.eq(input_tensor, other_tensor)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return result_bool.to(output_dtype)


def ttir_greater_than_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    result_bool = torch.gt(input_tensor, other_tensor)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return result_bool.to(output_dtype)


def ttir_typecast_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return input_tensor.to(output_dtype)


def ttir_log1p_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.log1p(input_tensor).to(output_dtype)


def ttir_log_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.log(input_tensor).to(output_dtype)


def ttir_maximum_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.maximum(input_tensor, other_tensor).to(output_dtype)


def ttir_multiply_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.multiply(input_tensor, other_tensor).to(output_dtype)


def ttir_add_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.add(input_tensor, other_tensor).to(output_dtype)


def ttir_sigmoid_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.sigmoid(input_tensor).to(output_dtype)


def ttir_hardsigmoid_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.nn.functional.hardsigmoid(input_tensor).to(output_dtype)


def ttir_subtract_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.subtract(input_tensor, other_tensor).to(output_dtype)


def ttir_tanh_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.tanh(input_tensor).to(output_dtype)


def ttir_rsqrt_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.rsqrt(input_tensor).to(output_dtype)


def ttir_neg_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.neg(input_tensor).to(output_dtype)


def ttir_where_golden(
    condition: GoldenMapTensor,
    x: GoldenMapTensor,
    y: GoldenMapTensor,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.where(condition, x, y).to(output_dtype)


def ttir_abs_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.abs(input_tensor).to(output_dtype)


def ttir_erf_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.erf(input_tensor).to(output_dtype)


def ttir_floor_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.floor(input_tensor).to(output_dtype)


def ttir_exp_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.exp(input_tensor).to(output_dtype)


def ttir_sort_golden(
    input_tensor: GoldenMapTensor,
    dim_attr: IntegerAttr,
    descending_attr: BoolAttr,
    stable_attr: BoolAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    dim = unpack_mlir_attr(dim_attr)
    descending = unpack_mlir_attr(descending_attr)
    stable = unpack_mlir_attr(stable_attr)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)

    values, indices = torch.sort(
        input_tensor, dim=dim, descending=descending, stable=stable
    )
    return values.to(output_dtype), indices.to(torch.int64)


def ttir_to_layout_golden(
    input_tensor: GoldenMapTensor, output_ranked_tensor_type: RankedTensorType
) -> GoldenMapTensor:
    casted_type = ttcore.ir.TileType.maybe_downcast(
        output_ranked_tensor_type.element_type
    )

    if casted_type:
        output_dtype = mlir_datatype_to_torch_dtype(casted_type.data_type)
    else:
        output_dtype = mlir_type_to_torch_dtype(output_ranked_tensor_type.element_type)

    output_tensor = input_tensor.clone()
    return output_tensor.to(output_dtype)


def ttir_all_gather_golden(
    input: GoldenMapTensor,
    all_gather_dim_attr: IntegerAttr,
    cluster_axis_attr: IntegerAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    all_gather_dim = unpack_mlir_attr(all_gather_dim_attr)
    cluster_axis = unpack_mlir_attr(cluster_axis_attr)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)

    output_shards = [None] * len(input.shard_map)
    grouped_shards = input.group_by_axis(cluster_axis)
    for group in grouped_shards:
        gathered_tensor = torch.cat(list(group.values()), dim=all_gather_dim)
        for id in group.keys():
            output_shards[id] = gathered_tensor.clone().to(output_dtype)
    return GoldenMapTensor(
        {i: t for i, t in enumerate(output_shards)}, input.mesh_shape
    )


def ttir_mesh_shard_golden(
    input: GoldenMapTensor,
    shard_type_attr: ttcore.ir.MeshShardTypeAttr,
    shard_direction_attr: ttcore.ir.MeshShardDirectionAttr,
    shard_shape_attr: DenseI64ArrayAttr,
    shard_dims_attr: DenseI64ArrayAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    mesh_shape = input.mesh_shape
    shard_type = ttcore.ir.MeshShardTypeAttr.maybe_downcast(shard_type_attr).value
    shard_direction = ttcore.ir.MeshShardDirectionAttr.maybe_downcast(
        shard_direction_attr
    ).value
    shard_shape = unpack_mlir_attr(shard_shape_attr)
    shard_dims = unpack_mlir_attr(shard_dims_attr)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)

    if shard_direction == ttcore.ir.MeshShardDirection.FullToShard:
        if shard_type == ttcore.ir.MeshShardType.Replicate:
            shard_dims = [None] * len(mesh_shape)
        return apply_sharding(input, mesh_shape, shard_dims)
    elif shard_direction == ttcore.ir.MeshShardDirection.ShardToFull:
        if shard_type == ttcore.ir.MeshShardType.Replicate:
            return apply_unsharding(input, [1], [1])
        return apply_unsharding(input, mesh_shape, shard_dims)


reduce_mapping = {
    ttcore.ir.ReduceType.Sum: lambda xs: torch.sum(torch.stack(xs), 0),
    ttcore.ir.ReduceType.Mean: lambda xs: torch.mean(torch.stack(xs), 0),
    ttcore.ir.ReduceType.Max: lambda xs: torch.amax(torch.stack(xs), 0),
    ttcore.ir.ReduceType.Min: lambda xs: torch.amin(torch.stack(xs), 0),
    ttcore.ir.ReduceType.Std: lambda xs: torch.std(torch.stack(xs), 0),
    ttcore.ir.ReduceType.Var: lambda xs: torch.var(torch.stack(xs), 0),
}


def ttir_all_reduce_golden(
    input: GoldenMapTensor,
    reduce_type_attr: ttcore.ir.ReduceTypeAttr,
    cluster_axis_attr: IntegerAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    reduce_type = ttcore.ir.ReduceTypeAttr.maybe_downcast(reduce_type_attr).value
    cluster_axis = unpack_mlir_attr(cluster_axis_attr)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)

    output_shards = [None] * len(input.shard_map)
    grouped_shards = input.group_by_axis(cluster_axis)
    for group in grouped_shards:
        group_tensors = list(group.values())
        reduced_tensor = reduce_mapping[reduce_type](group_tensors)
        for id in group.keys():
            output_shards[id] = reduced_tensor.clone().to(output_dtype)
    return GoldenMapTensor(
        {i: t for i, t in enumerate(output_shards)}, input.mesh_shape
    )


def ttir_reduce_scatter_golden(
    input: GoldenMapTensor,
    reduce_type_attr: ttcore.ir.ReduceTypeAttr,
    scatter_dim_attr: IntegerAttr,
    cluster_axis_attr: IntegerAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    reduce_type = ttcore.ir.ReduceTypeAttr.maybe_downcast(reduce_type_attr).value
    cluster_axis = unpack_mlir_attr(cluster_axis_attr)
    scatter_dim = unpack_mlir_attr(scatter_dim_attr)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)

    output_shards = [None] * len(input.shard_map)
    grouped_shards = input.group_by_axis(cluster_axis)
    for group in grouped_shards:
        group_tensors = list(group.values())
        reduced_tensor = reduce_mapping[reduce_type](group_tensors)
        scattered_tensor = torch.chunk(reduced_tensor, len(group), dim=scatter_dim)
        for index, id in enumerate(group.keys()):
            output_shards[id] = scattered_tensor[index].clone().to(output_dtype)
    return GoldenMapTensor(
        {i: t for i, t in enumerate(output_shards)}, input.mesh_shape
    )


def ttir_collective_permute_golden(
    input: GoldenMapTensor,
    source_target_pairs_attr: I64ElementsAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    source_target_pairs = unpack_mlir_attr(source_target_pairs_attr)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)

    output_shards = [torch.zeros_like(shard) for shard in input.shard_map.values()]
    for target_pairs in source_target_pairs:
        src, tgt = target_pairs
        output_shards[tgt] = input.shard_at(src).clone().to(output_dtype)
    return GoldenMapTensor(
        {i: t for i, t in enumerate(output_shards)}, input.mesh_shape
    )


def ttir_collective_broadcast_golden(
    input: GoldenMapTensor,
    replica_groups_attr: I64ElementsAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    replica_groups = unpack_mlir_attr(replica_groups_attr)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)

    output_shards = [None] * len(input.shard_map)
    for group in replica_groups:
        for device in group:
            output_shards[device] = input.shard_at(group[0]).clone().to(output_dtype)
    return GoldenMapTensor(
        {i: t for i, t in enumerate(output_shards)}, input.mesh_shape
    )


def ttir_all_to_all_golden(
    input: GoldenMapTensor,
    split_dim_attr: IntegerAttr,
    concat_dim_attr: IntegerAttr,
    split_count_attr: IntegerAttr,
    replica_groups_attr: I64ElementsAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    split_dim = unpack_mlir_attr(split_dim_attr)
    concat_dim = unpack_mlir_attr(concat_dim_attr)
    split_count = unpack_mlir_attr(split_count_attr)
    replica_groups = unpack_mlir_attr(replica_groups_attr)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)

    output_shards = [None] * len(input.shard_map)
    for group in replica_groups:
        splits_per_src: List[Tuple[torch.Tensor, ...]] = [
            torch.chunk(input.shard_at(dev_id), split_count, dim=split_dim)
            for dev_id in group
        ]
        for dst_idx in range(split_count):
            output_shards[group[dst_idx]] = (
                torch.cat(
                    [
                        splits_per_src[src_idx][dst_idx]
                        for src_idx in range(split_count)
                    ],
                    dim=concat_dim,
                )
                .clone()
                .to(output_dtype)
            )
    return GoldenMapTensor(
        {i: t for i, t in enumerate(output_shards)}, input.mesh_shape
    )


def ttir_isfinite_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.isfinite(input_tensor).to(dtype)


def ttir_embedding_backward_golden(
    indices_tensor: GoldenMapTensor,
    weight_tensor: GoldenMapTensor,
    in_gradient_tensor: GoldenMapTensor,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    dtype = mlir_type_to_torch_dtype(output_type_mlir)
    # Get the shape of the weight tensor (num_embeddings, embedding_dim)
    num_embeddings = weight_tensor.size()[0]
    embedding_dim = weight_tensor.size()[1]

    def compute_embedding_backward(
        indices_shard: torch.Tensor,
        in_gradient_shard: torch.Tensor,
    ) -> torch.Tensor:
        # Initialize output gradient with zeros
        grad_weight = torch.zeros(
            num_embeddings, embedding_dim, dtype=in_gradient_shard.dtype
        )

        # Flatten indices and gradients for easier accumulation
        indices_flat = indices_shard.to(torch.int64).flatten()
        grad_flat = in_gradient_shard.reshape(-1, embedding_dim)

        # Accumulate gradients at the corresponding indices
        grad_weight.index_add_(0, indices_flat, grad_flat)

        return grad_weight.to(dtype)

    # Apply the computation shard-wise and return a GoldenMapTensor
    result_shards = {}
    for device_id in indices_tensor.shard_map.keys():
        indices_shard = indices_tensor.shard_map[device_id]
        in_gradient_shard = in_gradient_tensor.shard_map[device_id]
        result_shards[device_id] = compute_embedding_backward(
            indices_shard, in_gradient_shard
        )

    return GoldenMapTensor(result_shards, indices_tensor.mesh_shape)


def ttir_concatenate_heads_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    # Input: [batch, num_heads, seq_len, head_dim]
    # Permute to: [batch, seq_len, num_heads, head_dim]
    permuted = input_tensor.permute(0, 2, 1, 3)
    # Reshape to: [batch, seq_len, num_heads * head_dim]
    batch, seq_len, num_heads, head_dim = permuted.shape
    return permuted.reshape(batch, seq_len, num_heads * head_dim)


def ttir_topk_golden(
    input_tensor: GoldenMapTensor,
    k_attr: IntegerAttr,
    dim_attr: IntegerAttr,
    largest_attr: BoolAttr,
    sorted_attr: BoolAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    k = unpack_mlir_attr(k_attr)
    dim = unpack_mlir_attr(dim_attr)
    largest = unpack_mlir_attr(largest_attr)
    sorted = unpack_mlir_attr(sorted_attr)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)

    values, indices = torch.topk(
        input_tensor, k=k, dim=dim, largest=largest, sorted=sorted
    )

    return values.to(output_dtype), indices.to(torch.uint16)


################ StableHLO Op Golden Functions ###############


def stablehlo_add_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.add(input_tensor, other_tensor).to(output_dtype)


def stablehlo_and_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    if output_dtype == torch.bool:
        result_bool = torch.logical_and(input_tensor, other_tensor)
        return result_bool.to(input_tensor.dtype)
    else:
        return torch.bitwise_and(input_tensor, other_tensor).to(output_dtype)


def stablehlo_abs_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.abs(input_tensor).to(output_dtype)


def stablehlo_ceil_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.ceil(input_tensor).to(output_dtype)


def stablehlo_cosine_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.cos(input_tensor).to(output_dtype)


def stablehlo_exp_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.exp(input_tensor).to(output_dtype)


def stablehlo_floor_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.floor(input_tensor).to(output_dtype)


def stablehlo_divide_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.div(input_tensor, other_tensor).to(output_dtype)


def stablehlo_clamp_golden(
    min_tensor: GoldenMapTensor,
    operand_tensor: GoldenMapTensor,
    max_tensor: GoldenMapTensor,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.clamp(operand_tensor, min=min_tensor, max=max_tensor).to(output_dtype)


def stablehlo_concatenate_golden(
    input_tensors: Tuple[GoldenMapTensor, ...],
    dim_attr: IntegerAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    dim = unpack_mlir_attr(dim_attr)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.cat(input_tensors, dim=dim).to(output_dtype)


def stablehlo_constant_golden(
    value: DenseElementsAttr, mesh_shape_attr: DenseI32ArrayAttr
) -> GoldenMapTensor:
    shape = list(value.type.shape)
    dtype = mlir_type_to_torch_dtype(value.type.element_type)

    if value.is_splat:
        value = value.get_splat_value()
        torch_tensor = torch.full(shape, value.value, dtype=dtype)
    else:
        torch_tensor = torch.tensor(np.array(value), dtype=dtype).reshape(shape)

    mesh_shape = unpack_mlir_attr(mesh_shape_attr)
    result = torch_tensor.reshape(shape)
    return GoldenMapTensor(
        {i: result.clone() for i in range(mesh_shape[0] * mesh_shape[1])}, mesh_shape
    )


def stablehlo_iota_golden(
    iota_dimension_attr: IntegerAttr,
    output_shape_attr: DenseI64ArrayAttr,
    mesh_shape_attr: DenseI32ArrayAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    iota_dimension = unpack_mlir_attr(iota_dimension_attr)
    output_shape = unpack_mlir_attr(output_shape_attr)
    dtype = mlir_type_to_torch_dtype(output_type_mlir)

    dim_size = output_shape[iota_dimension]
    iota_values = torch.arange(0, dim_size, dtype=dtype)

    broadcast_shape = [1] * len(output_shape)
    broadcast_shape[iota_dimension] = dim_size
    iota_values = iota_values.reshape(broadcast_shape)

    result = iota_values.expand(output_shape).clone()

    mesh_shape = unpack_mlir_attr(mesh_shape_attr)
    return GoldenMapTensor(
        {i: result.clone() for i in range(mesh_shape[0] * mesh_shape[1])}, mesh_shape
    )


def stablehlo_dynamic_iota_golden(
    output_shape: GoldenMapTensor,
    iota_dimension_attr: IntegerAttr,
    mesh_shape_attr: DenseI32ArrayAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    iota_dimension = unpack_mlir_attr(iota_dimension_attr)
    dtype = mlir_type_to_torch_dtype(output_type_mlir)

    shape_tensor = output_shape.shard_map[0]
    shape_list = shape_tensor.tolist()

    dim_size = int(shape_list[iota_dimension])
    iota_values = torch.arange(0, dim_size, dtype=dtype)

    broadcast_shape = [1] * len(shape_list)
    broadcast_shape[iota_dimension] = dim_size
    iota_values = iota_values.reshape(broadcast_shape)

    full_shape = [int(s) for s in shape_list]
    result = iota_values.expand(full_shape).clone()

    mesh_shape = unpack_mlir_attr(mesh_shape_attr)
    return GoldenMapTensor(
        {i: result.clone() for i in range(mesh_shape[0] * mesh_shape[1])}, mesh_shape
    )


def stablehlo_batch_norm_grad_golden(
    operand: GoldenMapTensor,
    scale: GoldenMapTensor,
    mean: GoldenMapTensor,
    variance: GoldenMapTensor,
    grad_output: GoldenMapTensor,
    epsilon: FloatAttr,
    feature_index: IntegerAttr,
    operand_output_type_mlir: Type,
    scale_output_type_mlir: Type,
    offset_output_type_mlir: Type,
) -> Tuple[GoldenMapTensor, GoldenMapTensor, GoldenMapTensor]:
    epsilon = unpack_mlir_attr(epsilon)
    feature_index = unpack_mlir_attr(feature_index)
    operand_output_dtype = mlir_type_to_torch_dtype(operand_output_type_mlir)
    scale_output_dtype = mlir_type_to_torch_dtype(scale_output_type_mlir)
    offset_output_dtype = mlir_type_to_torch_dtype(offset_output_type_mlir)

    grad_operand_shards = {}
    grad_scale_shards = {}
    grad_offset_shards = {}

    for device_id in operand.shard_map.keys():
        operand_shard = operand.shard_map[device_id]
        scale_shard = scale.shard_map[device_id]
        mean_shard = mean.shard_map[device_id]
        variance_shard = variance.shard_map[device_id]
        grad_output_shard = grad_output.shard_map[device_id]

        ndim = operand_shard.ndim

        # Compute the dimensions to reduce over (all dims except feature_index)
        reduce_dims = [i for i in range(ndim) if i != feature_index]

        # Compute the number of elements per feature
        n = 1
        for dim in reduce_dims:
            n *= operand_shard.shape[dim]

        # Reshape scale, mean, variance to broadcast correctly
        broadcast_shape = [1] * ndim
        broadcast_shape[feature_index] = scale_shard.shape[0]
        scale_bc = scale_shard.reshape(broadcast_shape)
        mean_bc = mean_shard.reshape(broadcast_shape)
        variance_bc = variance_shard.reshape(broadcast_shape)

        # Compute standard deviation
        std = torch.sqrt(variance_bc + epsilon)

        # Normalized input
        x_norm = (operand_shard - mean_bc) / std

        # grad_offset: sum of grad_output over all dimensions except feature_index
        grad_offset_shard = grad_output_shard.sum(dim=reduce_dims)

        # grad_scale: sum of (grad_output * x_norm) over all dimensions except feature_index
        grad_scale_shard = (grad_output_shard * x_norm).sum(dim=reduce_dims)

        # grad_operand: more complex, involves the chain rule through normalization
        # grad_x = (1/std) * (grad_output - (1/n) * grad_offset_bc - (1/n) * x_norm * grad_scale_bc)
        # where grad_offset_bc and grad_scale_bc are broadcast versions
        grad_offset_bc = grad_offset_shard.reshape(broadcast_shape)
        grad_scale_bc = grad_scale_shard.reshape(broadcast_shape)

        grad_operand_shard = (
            scale_bc
            / std
            * (grad_output_shard - grad_offset_bc / n - x_norm * grad_scale_bc / n)
        )

        grad_operand_shards[device_id] = grad_operand_shard.to(operand_output_dtype)
        grad_scale_shards[device_id] = grad_scale_shard.to(scale_output_dtype)
        grad_offset_shards[device_id] = grad_offset_shard.to(offset_output_dtype)

    return (
        GoldenMapTensor(grad_operand_shards, operand.mesh_shape),
        GoldenMapTensor(grad_scale_shards, scale.mesh_shape),
        GoldenMapTensor(grad_offset_shards, scale.mesh_shape),
    )


def stablehlo_batch_norm_training_golden(
    operand: GoldenMapTensor,
    scale: GoldenMapTensor,
    offset: GoldenMapTensor,
    epsilon: FloatAttr,
    feature_index: IntegerAttr,
    output_type_mlir: Type,
    mean_output_type_mlir: Type,
    variance_output_type_mlir: Type,
) -> Tuple[GoldenMapTensor, GoldenMapTensor, GoldenMapTensor]:
    epsilon = unpack_mlir_attr(epsilon)
    feature_index = unpack_mlir_attr(feature_index)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    mean_output_dtype = mlir_type_to_torch_dtype(mean_output_type_mlir)
    variance_output_dtype = mlir_type_to_torch_dtype(variance_output_type_mlir)

    output_shards = {}
    batch_mean_shards = {}
    batch_var_shards = {}

    for device_id in operand.shard_map.keys():
        operand_shard = operand.shard_map[device_id]
        scale_shard = scale.shard_map[device_id]
        offset_shard = offset.shard_map[device_id]

        ndim = operand_shard.ndim

        # Compute the dimensions to reduce over (all dims except feature_index)
        reduce_dims = [i for i in range(ndim) if i != feature_index]

        # Compute batch mean and variance
        batch_mean_shard = torch.mean(operand_shard, dim=reduce_dims)
        batch_var_shard = torch.var(operand_shard, dim=reduce_dims, unbiased=False)

        # Reshape for broadcasting
        broadcast_shape = [1] * ndim
        broadcast_shape[feature_index] = scale_shard.shape[0]

        batch_mean_bc = batch_mean_shard.reshape(broadcast_shape)
        batch_var_bc = batch_var_shard.reshape(broadcast_shape)
        scale_bc = scale_shard.reshape(broadcast_shape)
        offset_bc = offset_shard.reshape(broadcast_shape)

        # Normalize: (x - mean) / sqrt(var + eps) * scale + offset
        std = torch.sqrt(batch_var_bc + epsilon)
        normalized = (operand_shard - batch_mean_bc) / std
        output_shard = normalized * scale_bc + offset_bc

        output_shards[device_id] = output_shard.to(output_dtype)
        batch_mean_shards[device_id] = batch_mean_shard.to(mean_output_dtype)
        batch_var_shards[device_id] = batch_var_shard.to(variance_output_dtype)

    return (
        GoldenMapTensor(output_shards, operand.mesh_shape),
        GoldenMapTensor(batch_mean_shards, scale.mesh_shape),
        GoldenMapTensor(batch_var_shards, scale.mesh_shape),
    )


def stablehlo_batch_norm_inference_golden(
    operand: GoldenMapTensor,
    scale: GoldenMapTensor,
    offset: GoldenMapTensor,
    mean: GoldenMapTensor,
    variance: GoldenMapTensor,
    epsilon: FloatAttr,
    feature_index: IntegerAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    epsilon = unpack_mlir_attr(epsilon)
    feature_index = unpack_mlir_attr(feature_index)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    output_shards = {}
    for device_id in operand.shard_map.keys():
        operand_shard = operand.shard_map[device_id]
        scale_shard = scale.shard_map[device_id]
        offset_shard = offset.shard_map[device_id]
        mean_shard = mean.shard_map[device_id]
        variance_shard = variance.shard_map[device_id]

        ndim = operand_shard.ndim

        # Reshape for broadcasting
        broadcast_shape = [1] * ndim
        broadcast_shape[feature_index] = scale_shard.shape[0]

        mean_bc = mean_shard.reshape(broadcast_shape)
        var_bc = variance_shard.reshape(broadcast_shape)
        scale_bc = scale_shard.reshape(broadcast_shape)
        offset_bc = offset_shard.reshape(broadcast_shape)

        # Normalize: (x - mean) / sqrt(var + eps) * scale + offset
        std = torch.sqrt(var_bc + epsilon)
        normalized = (operand_shard - mean_bc) / std
        output_shard = normalized * scale_bc + offset_bc

        output_shards[device_id] = output_shard.to(output_dtype)

    return GoldenMapTensor(output_shards, operand.mesh_shape)


def stablehlo_log_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.log(input_tensor).to(output_dtype)


def stablehlo_log1p_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.log1p(input_tensor).to(output_dtype)


def stablehlo_logistic_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.sigmoid(input_tensor).to(output_dtype)


def stablehlo_neg_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.neg(input_tensor).to(output_dtype)


def stablehlo_reshape_golden(
    input_tensor: GoldenMapTensor, shape_attr: ArrayAttr, output_type_mlir: Type
) -> GoldenMapTensor:
    shape = unpack_mlir_attr(shape_attr)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.reshape(input_tensor, shape).clone().to(output_dtype)


def stablehlo_rsqrt_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.rsqrt(input_tensor).to(output_dtype)


def stablehlo_slice_golden(
    input_tensor: GoldenMapTensor,
    start_indices_attr: DenseI64ArrayAttr,
    limit_indices_attr: DenseI64ArrayAttr,
    strides_attr: DenseI64ArrayAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    start_indices = unpack_mlir_attr(start_indices_attr)
    limit_indices = unpack_mlir_attr(limit_indices_attr)
    strides = unpack_mlir_attr(strides_attr)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)

    slices = []
    for i in range(len(start_indices)):
        start = start_indices[i] if i < len(start_indices) else 0
        end = limit_indices[i] if i < len(limit_indices) else input_tensor.size(i)
        stride = strides[i] if i < len(strides) else 1
        slices.append(slice(start, end, stride))

    shard_map = {}
    for device_id, shard in input_tensor.shard_map.items():
        shard_map[device_id] = shard[tuple(slices)]
        shard_map[device_id] = shard_map[device_id].to(output_dtype)

    return GoldenMapTensor(shard_map, input_tensor.mesh_shape)


def stablehlo_get_dimension_size_golden(
    input_tensor: GoldenMapTensor,
    dimension_attr: IntegerAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    dimension = unpack_mlir_attr(dimension_attr)
    output_tensor = input_tensor.clone()

    for device_id, shard in input_tensor.shard_map.items():
        shard = torch.tensor(
            input_tensor.shard_at(device_id).size(dimension), dtype=torch.int32
        )
        output_tensor.shard_map[device_id] = shard

    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return output_tensor.to(output_dtype)


def stablehlo_sine_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.sin(input_tensor).to(output_dtype)


def stablehlo_sqrt_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.sqrt(input_tensor).to(output_dtype)


def stablehlo_tan_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.tan(input_tensor).to(output_dtype)


def stablehlo_tanh_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.tanh(input_tensor).to(output_dtype)


def stablehlo_transpose_golden(
    input_tensor: GoldenMapTensor,
    permutation: DenseI64ArrayAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    permutation = unpack_mlir_attr(permutation)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.permute(input_tensor, tuple(permutation)).to(output_dtype)


def stablehlo_select_golden(
    pred_tensor: GoldenMapTensor,
    on_true_tensor: GoldenMapTensor,
    on_false_tensor: GoldenMapTensor,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    pred_bool = pred_tensor.to(torch.bool)
    return torch.where(pred_bool, on_true_tensor, on_false_tensor).to(output_dtype)


def stablehlo_reverse_golden(
    input_tensor: GoldenMapTensor,
    dimensions_attr: DenseI64ArrayAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    dims = unpack_mlir_attr(dimensions_attr)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.flip(input_tensor, dims).to(output_dtype)


def stablehlo_maximum_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.maximum(input_tensor, other_tensor).to(output_dtype)


def stablehlo_minimum_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.minimum(input_tensor, other_tensor).to(output_dtype)


def stablehlo_multiply_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.multiply(input_tensor, other_tensor).to(output_dtype)


def stablehlo_pow_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.pow(input_tensor, other_tensor).to(output_dtype)


def stablehlo_subtract_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.subtract(input_tensor, other_tensor).to(output_dtype)


def stablehlo_shift_right_logical_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    shifted = logical_right_shift_golden(input_tensor, other_tensor)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return shifted.to(output_dtype)


# The following golden implementation is taken from the op spec: https://openxla.org/stablehlo/spec#dynamic_update_slice
def stablehlo_dynamic_update_slice_golden(
    input_tensor: GoldenMapTensor,
    update_tensor: GoldenMapTensor,
    start_indices: List[GoldenMapTensor],
    output_type_mlir: Type,
) -> GoldenMapTensor:
    def clamp(min_val, x, max_val):
        return max(min_val, min(x, max_val))

    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)

    result_shard_map = {}
    for device_id in input_tensor.shard_map.keys():
        input_shard = input_tensor.shard_map[device_id]
        update_shard = update_tensor.shard_map[device_id]
        start_indices_shard = [
            idx_tensor.shard_map[device_id] for idx_tensor in start_indices
        ]
        result_shard = input_shard.clone()

        input_shape = input_shard.shape
        update_shape = update_shard.shape
        input_rank = len(input_shape)

        # adjusted_start_indices = clamp(0, start_indices, shape(operand) - shape(update))
        adjusted_start_indices = tuple(
            clamp(0, int(start_indices_shard[d]), input_shape[d] - update_shape[d])
            for d in range(input_rank)
        )

        for result_index in itertools.product(*[range(s) for s in input_shape]):
            # update_index = result_index - adjusted_start_indices
            update_index = tuple(
                result_index[d] - adjusted_start_indices[d] for d in range(input_rank)
            )

            # 0 <= update_index < shape(update)
            in_update = all(
                0 <= update_index[d] < update_shape[d] for d in range(input_rank)
            )

            if in_update:
                result_shard[result_index] = update_shard[update_index]
            else:
                result_shard[result_index] = input_shard[result_index]

            result_shard_map[device_id] = result_shard

    return GoldenMapTensor(result_shard_map, input_tensor.mesh_shape).to(output_dtype)


def stablehlo_all_gather_golden(
    input: GoldenMapTensor,
    all_gather_dim_attr: IntegerAttr,
    replica_groups_attr: DenseElementsAttr,
) -> GoldenMapTensor:
    all_gather_dim = unpack_mlir_attr(all_gather_dim_attr)
    replica_groups = unpack_mlir_attr(replica_groups_attr)

    output_shards = [None] * len(input.shard_map)
    for group in replica_groups:
        gathered_tensor = torch.cat(
            [input.shard_at(dev_id) for dev_id in group], dim=all_gather_dim
        )
        for id in group:
            output_shards[id] = gathered_tensor.clone()
    return GoldenMapTensor(
        {i: t for i, t in enumerate(output_shards)}, input.mesh_shape
    )


def stablehlo_all_reduce_golden(
    input: GoldenMapTensor,
    replica_groups_attr: DenseElementsAttr,
) -> GoldenMapTensor:
    replica_groups = unpack_mlir_attr(replica_groups_attr)
    raise NotImplementedError("stablehlo_all_reduce_golden is not implemented yet.")


def stablehlo_all_to_all_golden(
    input: GoldenMapTensor,
    split_dim_attr: IntegerAttr,
    concat_dim_attr: IntegerAttr,
    split_count_attr: IntegerAttr,
    replica_groups_attr: DenseElementsAttr,
) -> GoldenMapTensor:
    split_dim = unpack_mlir_attr(split_dim_attr)
    concat_dim = unpack_mlir_attr(concat_dim_attr)
    split_count = unpack_mlir_attr(split_count_attr)
    replica_groups = unpack_mlir_attr(replica_groups_attr)
    raise NotImplementedError("stablehlo_all_to_all_golden is not implemented yet.")


def stablehlo_collective_broadcast_golden(
    input: GoldenMapTensor,
    replica_groups_attr: DenseElementsAttr,
) -> GoldenMapTensor:
    replica_groups = unpack_mlir_attr(replica_groups_attr)
    raise NotImplementedError(
        "stablehlo_collective_broadcast_golden is not implemented yet."
    )


def stablehlo_collective_permute_golden(
    input: GoldenMapTensor,
    source_target_pairs_attr: DenseElementsAttr,
) -> GoldenMapTensor:
    source_target_pairs = unpack_mlir_attr(source_target_pairs_attr)
    raise NotImplementedError(
        "stablehlo_collective_permute_golden is not implemented yet."
    )


def stablehlo_reduce_scatter_golden(
    input: GoldenMapTensor,
    scatter_dim_attr: IntegerAttr,
    replica_groups_attr: DenseElementsAttr,
) -> GoldenMapTensor:
    scatter_dim = unpack_mlir_attr(scatter_dim_attr)
    replica_groups = unpack_mlir_attr(replica_groups_attr)
    raise NotImplementedError("stablehlo_reduce_scatter_golden is not implemented yet.")


def stablehlo_pad_golden(
    input_tensor: GoldenMapTensor,
    value: GoldenMapTensor,
    edge_padding_low: DenseI64ArrayAttr,
    edge_padding_high: DenseI64ArrayAttr,
    interior_padding: DenseI64ArrayAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    padding_low = unpack_mlir_attr(edge_padding_low)
    padding_high = unpack_mlir_attr(edge_padding_high)
    interior = unpack_mlir_attr(interior_padding)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)

    # padding value is a 0-rank tensor, but torch expects the
    # value to be python scalar (float). Hence, extract the value by first
    # converting golden tensor to torch tensors, then
    # obtain the pad_value
    pad_value = value.contiguous().shard_map[0].item()

    rank = len(padding_low)
    assert len(padding_high) == rank
    assert len(interior) == rank

    # Reverse the padding values as torch expects the
    # padding in the reverse order (i.e from last dimension moving
    # leftwards).
    # [low_last, high_last, low_prev, high_prev, ..., low0, high0]
    golden_padding = []
    for d in reversed(range(rank)):
        golden_padding.extend([padding_low[d], padding_high[d]])

    return torch.nn.functional.pad(
        input_tensor, pad=golden_padding, mode="constant", value=pad_value
    ).to(output_dtype)


def stablehlo_reduce_window_golden(
    input_tensor: GoldenMapTensor,
    init_value: GoldenMapTensor,
    window_dimensions: DenseI64ArrayAttr,
    window_strides: Optional[DenseI64ArrayAttr],
    base_dilations: Optional[DenseI64ArrayAttr],
    window_dilations: Optional[DenseI64ArrayAttr],
    padding: Optional[DenseI64ArrayAttr],
    output_type_mlir: Type,
    body: str = "add",
) -> GoldenMapTensor:
    """
    Golden function for stablehlo.reduce_window operation.

    Applies a reduction function to sliding windows over the input tensor.
    """
    window_dims = unpack_mlir_attr(window_dimensions)
    w_strides = (
        unpack_mlir_attr(window_strides)
        if window_strides is not None
        else [1] * len(window_dims)
    )
    b_dilations = (
        unpack_mlir_attr(base_dilations)
        if base_dilations is not None
        else [1] * len(window_dims)
    )
    w_dilations = (
        unpack_mlir_attr(window_dilations)
        if window_dilations is not None
        else [1] * len(window_dims)
    )

    if padding is not None:
        padding_attr = unpack_mlir_attr(padding)
        if isinstance(padding_attr, np.ndarray):
            if padding_attr.ndim == 2:
                pad_2d = [[int(p[0]), int(p[1])] for p in padding_attr]
            else:
                rank = len(window_dims)
                pad_2d = [
                    [int(padding_attr[i * 2]), int(padding_attr[i * 2 + 1])]
                    for i in range(rank)
                ]
        elif isinstance(padding_attr, (list, tuple)) and len(padding_attr) > 0:
            if isinstance(padding_attr[0], (list, tuple)):
                pad_2d = [list(p) for p in padding_attr]
            else:
                rank = len(window_dims)
                pad_2d = [
                    [padding_attr[i * 2], padding_attr[i * 2 + 1]] for i in range(rank)
                ]
        else:
            pad_2d = [[0, 0] for _ in range(len(window_dims))]
    else:
        pad_2d = [[0, 0] for _ in range(len(window_dims))]

    if hasattr(output_type_mlir, "element_type"):
        output_dtype = mlir_type_to_torch_dtype(output_type_mlir.element_type)
    else:
        output_dtype = mlir_type_to_torch_dtype(output_type_mlir)

    init_scalar = init_value.contiguous().shard_map[0].item()

    output_shards = {}
    for device_id in input_tensor.shard_map.keys():
        input_shard = input_tensor.shard_map[device_id]
        output_shard = _reduce_window_impl(
            input_shard,
            init_scalar,
            window_dims,
            w_strides,
            b_dilations,
            w_dilations,
            pad_2d,
            body,
        )
        output_shards[device_id] = output_shard.to(output_dtype)

    return GoldenMapTensor(output_shards, input_tensor.mesh_shape)


def _reduce_window_impl(
    input_tensor: torch.Tensor,
    init_value: Union[float, int],
    window_dimensions: List[int],
    window_strides: List[int],
    base_dilations: List[int],
    window_dilations: List[int],
    padding: List[List[int]],
    body: str,
) -> torch.Tensor:
    """
    Implementation of reduce_window computation for golden reference.
    """
    input_shape = list(input_tensor.shape)
    rank = len(input_shape)

    if any(d != 1 for d in base_dilations):
        dilated_shape = [
            (s - 1) * d + 1 if s > 0 else 0 for s, d in zip(input_shape, base_dilations)
        ]
        dilated_input = torch.full(dilated_shape, init_value, dtype=input_tensor.dtype)
        slices = tuple(slice(None, None, d) for d in base_dilations)
        dilated_input[slices] = input_tensor
        input_tensor = dilated_input
        input_shape = list(input_tensor.shape)

    pad_list = []
    for i in range(rank - 1, -1, -1):
        pad_list.extend([padding[i][0], padding[i][1]])
    if any(p != 0 for p in pad_list):
        input_tensor = torch.nn.functional.pad(input_tensor, pad_list, value=init_value)
        input_shape = list(input_tensor.shape)

    output_shape = []
    for i in range(rank):
        dilated_window = (
            (window_dimensions[i] - 1) * window_dilations[i] + 1
            if window_dimensions[i] > 0
            else 0
        )
        if input_shape[i] == 0 or dilated_window > input_shape[i]:
            output_dim = 0
        else:
            output_dim = ((input_shape[i] - dilated_window) // window_strides[i]) + 1
        output_shape.append(output_dim)

    output_tensor = torch.full(output_shape, init_value, dtype=input_tensor.dtype)

    if all(d > 0 for d in output_shape):
        output_indices = itertools.product(*[range(s) for s in output_shape])
        window_offsets = list(itertools.product(*[range(s) for s in window_dimensions]))

        for idx in output_indices:
            window_elements = []
            for window_idx in window_offsets:
                input_idx = tuple(
                    idx[d] * window_strides[d] + window_idx[d] * window_dilations[d]
                    for d in range(rank)
                )
                if all(0 <= input_idx[d] < input_shape[d] for d in range(rank)):
                    window_elements.append(input_tensor[input_idx].item())

            if window_elements:
                if body == "add":
                    result = sum(window_elements)
                elif body == "max":
                    result = max(window_elements)
                else:
                    result = init_value
                output_tensor[idx] = result

    return output_tensor


def stablehlo_convolution_golden(
    lhs: GoldenMapTensor,
    rhs: GoldenMapTensor,
    window_strides_attr: DenseI64ArrayAttr,
    padding_attr: DenseI64ArrayAttr,
    lhs_dilation_attr: DenseI64ArrayAttr,
    rhs_dilation_attr: DenseI64ArrayAttr,
    window_reversal_attr: DenseBoolArrayAttr,
    dimension_numbers_attr: Attribute,
    feature_group_count_attr: IntegerAttr,
    batch_group_count_attr: IntegerAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    input_tensor = lhs.clone()
    weight = rhs.clone()
    window_strides = unpack_mlir_attr(window_strides_attr)
    lhs_dilation = unpack_mlir_attr(lhs_dilation_attr)
    rhs_dilation = unpack_mlir_attr(rhs_dilation_attr)
    feature_group_count = unpack_mlir_attr(feature_group_count_attr)
    batch_group_count = unpack_mlir_attr(batch_group_count_attr)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)

    # Parse dimension numbers using StableHLO's ConvDimensionNumbers
    dim_numbers = stablehlo.ConvDimensionNumbers.maybe_downcast(dimension_numbers_attr)

    # Input layout
    input_batch = dim_numbers.input_batch_dimension
    input_feature = dim_numbers.input_feature_dimension
    input_spatial_dimensions = list(dim_numbers.input_spatial_dimensions)

    # Output layout
    output_batch = dim_numbers.output_batch_dimension
    output_feature = dim_numbers.output_feature_dimension
    output_spatial_dimensions = list(dim_numbers.output_spatial_dimensions)

    # Kernel layout
    kernel_output_feature = dim_numbers.kernel_output_feature_dimension
    kernel_input_feature = dim_numbers.kernel_input_feature_dimension
    kernel_spatial_dimensions = list(dim_numbers.kernel_spatial_dimensions)

    # Permute input tensor to NCHW format (PyTorch expects this)
    # Current layout is defined by dimension positions
    current_layout = [None] * input_tensor.ndim
    current_layout[input_batch] = 0  # batch goes to position 0
    current_layout[input_feature] = 1  # feature goes to position 1
    for i, spatial_dim in enumerate(input_spatial_dimensions):
        current_layout[spatial_dim] = 2 + i  # spatial dims go to positions 2, 3, ...

    if current_layout != list(range(input_tensor.ndim)):
        permutation = [current_layout.index(i) for i in range(input_tensor.ndim)]
        input_tensor = input_tensor.permute(permutation)

    # Compute output permutation (from NCHW back to output layout)
    output_permutation = [None] * (len(output_spatial_dimensions) + 2)
    output_permutation[output_batch] = 0
    output_permutation[output_feature] = 1
    for i, spatial_dim in enumerate(output_spatial_dimensions):
        output_permutation[spatial_dim] = 2 + i

    # Permute weight tensor to PyTorch format [output_channels, input_channels, H, W]
    weight_layout = [None] * weight.ndim
    weight_layout[kernel_output_feature] = 0
    weight_layout[kernel_input_feature] = 1
    for i, spatial_dim in enumerate(kernel_spatial_dimensions):
        weight_layout[spatial_dim] = 2 + i

    if weight_layout != list(range(weight.ndim)):
        weight_permutation = [weight_layout.index(i) for i in range(weight.ndim)]
        weight = weight.permute(weight_permutation)

    # Extract stride (StableHLO uses spatial-only strides)
    stride = list(window_strides) if window_strides else [1, 1]

    # Extract dilation (rhs_dilation is weight dilation in StableHLO)
    dilation = list(rhs_dilation) if rhs_dilation else [1, 1]

    # Handle padding - StableHLO uses [[low_h, high_h], [low_w, high_w]] format
    padding = unpack_mlir_attr(padding_attr)
    # Handle 2D array format from DenseElementsAttr (numpy array with shape (2, 2))
    if hasattr(padding, "shape") and len(padding.shape) == 2:
        # 2D format: [[low_h, high_h], [low_w, high_w]]
        low_h, high_h = int(padding[0, 0]), int(padding[0, 1])
        low_w, high_w = int(padding[1, 0]), int(padding[1, 1])
        if low_h == high_h and low_w == high_w:
            torch_padding = [low_h, low_w]
        else:
            # Asymmetric padding - manually pad the input
            # PyTorch F.pad expects [left, right, top, bottom] for 4D input
            input_tensor = torch.nn.functional.pad(
                input_tensor, [low_w, high_w, low_h, high_h], mode="constant", value=0
            )
            torch_padding = [0, 0]
    elif len(padding) == 4:
        # Flattened format: [low_h, high_h, low_w, high_w]
        low_h, high_h, low_w, high_w = padding
        if low_h == high_h and low_w == high_w:
            torch_padding = [low_h, low_w]
        else:
            # Asymmetric padding - manually pad the input
            # PyTorch F.pad expects [left, right, top, bottom] for 4D input
            input_tensor = torch.nn.functional.pad(
                input_tensor, [low_w, high_w, low_h, high_h], mode="constant", value=0
            )
            torch_padding = [0, 0]
    elif len(padding) == 2:
        torch_padding = [int(padding[0]), int(padding[1])]
    else:
        torch_padding = [0, 0]

    # Ensure matching dtypes
    if input_tensor.dtype != weight.dtype:
        weight = weight.to(input_tensor.dtype)

    # Handle lhs_dilation (input dilation) - used for transposed convolutions
    if lhs_dilation and any(d > 1 for d in lhs_dilation):
        result = torch.nn.functional.conv_transpose2d(
            input_tensor,
            weight,
            bias=None,
            stride=tuple(lhs_dilation),  # lhs_dilation is the upsampling factor
            padding=tuple(torch_padding),
            output_padding=0,
            dilation=tuple(dilation),
            groups=feature_group_count,
        )
    else:
        result = torch.nn.functional.conv2d(
            input_tensor,
            weight,
            bias=None,
            stride=tuple(stride),
            padding=tuple(torch_padding),
            dilation=tuple(dilation),
            groups=feature_group_count,
        )

    # Permute output back to expected layout if needed
    if output_permutation != list(range(result.ndim)):
        result = result.permute(output_permutation)

    return result.to(output_dtype)


################ SDY Op Golden Functions ###############


def sdy_sharding_constraint_golden(
    input: GoldenMapTensor,
) -> GoldenMapTensor:
    return input.clone()


def sdy_reshard_golden(input: GoldenMapTensor) -> GoldenMapTensor:
    return input.clone()


def sdy_all_gather_golden(
    input: GoldenMapTensor,
) -> GoldenMapTensor:
    return input.clone()


################ TTNN Op Golden Functions ###############


def ttnn_abs_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.abs(input_tensor).to(dtype)


def ttnn_cbrt_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    dtype = mlir_type_to_torch_dtype(output_type_mlir)
    golden_sign = torch.sign(input_tensor)
    golden_cbrt = torch.pow(torch.abs(input_tensor), 1 / 3)
    return torch.mul(golden_sign, golden_cbrt).to(dtype)


def ttnn_ceil_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.ceil(input_tensor).to(dtype)


def ttnn_cos_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.cos(input_tensor).to(dtype)


def ttnn_erf_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.erf(input_tensor).to(dtype)


def ttnn_erfc_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.erfc(input_tensor).to(dtype)


def ttnn_exp_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.exp(input_tensor).to(dtype)


def ttnn_floor_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.floor(input_tensor).to(dtype)


def ttnn_gelu_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.nn.functional.gelu(input_tensor).to(dtype)


def ttnn_isfinite_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.isfinite(input_tensor).to(dtype)


def ttnn_neg_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.neg(input_tensor).to(dtype)


def ttnn_tan_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.tan(input_tensor).to(dtype)


def ttnn_atan_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.atan(input_tensor).to(dtype)


def ttnn_tanh_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.tanh(input_tensor).to(dtype)


def ttnn_reciprocal_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.reciprocal(input_tensor).to(dtype)


def ttnn_relu_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.nn.functional.relu(input_tensor).to(dtype)


def ttnn_relu6_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.clamp(input_tensor, min=0, max=6).to(dtype)


def ttnn_rsqrt_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.rsqrt(input_tensor).to(dtype)


def ttnn_sigmoid_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.sigmoid(input_tensor).to(dtype)


def ttnn_sign_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.sign(input_tensor).to(dtype)


def ttnn_silu_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.nn.functional.silu(input_tensor).to(dtype)


def ttnn_sin_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.sin(input_tensor).to(dtype)


def ttnn_sqrt_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.sqrt(input_tensor).to(dtype)


def ttnn_typecast_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return input_tensor.to(dtype)


def ttnn_log_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.log(input_tensor).to(dtype)


def ttnn_log1p_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.log1p(input_tensor).to(dtype)


def ttnn_expm1_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.expm1(input_tensor).to(dtype)


def ttnn_add_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.add(input_tensor, other_tensor).to(output_dtype)


def ttnn_eq_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    result_bool = torch.eq(input_tensor, other_tensor)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return result_bool.to(output_dtype)


def ttnn_ne_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    result_bool = torch.ne(input_tensor, other_tensor)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return result_bool.to(output_dtype)


def ttnn_ge_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.ge(input_tensor, other_tensor).to(output_dtype)


def ttnn_gt_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    result_bool = torch.gt(input_tensor, other_tensor)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return result_bool.to(output_dtype)


def ttnn_le_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.le(input_tensor, other_tensor).to(output_dtype)


def ttnn_lt_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.lt(input_tensor, other_tensor).to(output_dtype)


def ttnn_logical_and_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.logical_and(input_tensor, other_tensor).to(output_dtype)


def ttnn_logical_or_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.logical_or(input_tensor, other_tensor).to(output_dtype)


def ttnn_logical_not_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.logical_not(input_tensor).to(output_dtype)


def ttnn_logical_xor_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.logical_xor(input_tensor, other_tensor).to(output_dtype)


def ttnn_logical_right_shift_golden(
    input_tensor: GoldenMapTensor, shift_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    input_int64 = input_tensor.to(torch.int64)
    shift_int64 = shift_tensor.to(torch.int64)
    input_unsigned = torch.bitwise_and(input_int64, 0xFFFFFFFF)
    result = torch.bitwise_right_shift(input_unsigned, shift_int64)
    return torch.bitwise_and(result, 0xFFFFFFFF).to(output_dtype)


def ttnn_logical_left_shift_golden(
    input_tensor: GoldenMapTensor, shift_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    input_int64 = input_tensor.to(torch.int64)
    shift_int64 = shift_tensor.to(torch.int64)
    input_unsigned = torch.bitwise_and(input_int64, 0xFFFFFFFF)
    result = torch.bitwise_left_shift(input_unsigned, shift_int64)
    return torch.bitwise_and(result, 0xFFFFFFFF).to(output_dtype)


def ttnn_bitwise_and_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.bitwise_and(input_tensor, other_tensor).to(output_dtype)


def ttnn_bitwise_or_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.bitwise_or(input_tensor, other_tensor).to(output_dtype)


def ttnn_bitwise_xor_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.bitwise_xor(input_tensor, other_tensor).to(output_dtype)


def ttnn_bitwise_not_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.bitwise_not(input_tensor).to(output_dtype)


def ttnn_minimum_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.minimum(input_tensor, other_tensor).to(output_dtype)


def ttnn_maximum_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.maximum(input_tensor, other_tensor).to(output_dtype)


def ttnn_multiply_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.multiply(input_tensor, other_tensor).to(output_dtype)


def ttnn_subtract_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.subtract(input_tensor, other_tensor).to(output_dtype)


def ttnn_remainder_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.remainder(input_tensor, other_tensor).to(output_dtype)


def ttnn_pow_tensor_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.pow(input_tensor, other_tensor).to(output_dtype)


def ttnn_divide_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.div(input_tensor, other_tensor).to(output_dtype)


def ttnn_atan2_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.atan2(input_tensor, other_tensor).to(output_dtype)


def ttnn_matmul_golden(
    input_tensor: GoldenMapTensor,
    other_tensor: GoldenMapTensor,
    transpose_a_attr: BoolAttr,
    transpose_b_attr: BoolAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    transpose_a = unpack_mlir_attr(transpose_a_attr)
    transpose_b = unpack_mlir_attr(transpose_b_attr)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    a = torch.transpose(input_tensor, -2, -1) if transpose_a else input_tensor
    b = torch.transpose(other_tensor, -2, -1) if transpose_b else other_tensor
    return torch.matmul(a, b).to(output_dtype)


def ttnn_linear_golden(
    input_tensor: GoldenMapTensor,
    other_tensor: GoldenMapTensor,
    bias_tensor: Optional[GoldenMapTensor],
    transpose_a_attr: BoolAttr,
    transpose_b_attr: BoolAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    transpose_a = unpack_mlir_attr(transpose_a_attr)
    transpose_b = unpack_mlir_attr(transpose_b_attr)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    a = torch.transpose(input_tensor, -2, -1) if transpose_a else input_tensor
    b = torch.transpose(other_tensor, -2, -1) if transpose_b else other_tensor
    output = torch.matmul(a, b)

    if bias_tensor is None:
        bias_tensor = torch.zeros(list(output.shape))

    bias_tensor = (
        torch.broadcast_to(bias_tensor, list(output.shape))
        if bias_tensor.shape != output.shape
        else bias_tensor
    )
    return torch.add(output, bias_tensor).to(output_dtype)


def ttnn_layer_norm_golden(
    input: GoldenMapTensor,
    weight: Optional[GoldenMapTensor],
    bias: Optional[GoldenMapTensor],
    normalized_shape: ArrayAttr,
    epsilon: FloatAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    normalized_shape = unpack_mlir_attr(normalized_shape)
    epsilon = unpack_mlir_attr(epsilon)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    input_float = input.float()

    return torch.nn.functional.layer_norm(
        input_float,
        normalized_shape=normalized_shape,
        weight=weight,
        bias=bias,
        eps=epsilon,
    ).to(output_dtype)


def ttnn_concat_golden(
    input_tensors: List[GoldenMapTensor], dim_attr: IntegerAttr, output_type_mlir: Type
) -> GoldenMapTensor:
    dim = unpack_mlir_attr(dim_attr)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    if isinstance(input_tensors, tuple):
        return torch.concat(input_tensors, dim=dim).to(output_dtype)
    else:
        return torch.concat([input_tensors], dim=dim).to(output_dtype)


def ttnn_repeat_golden(
    input_tensor: GoldenMapTensor,
    repeat_dims_attr: Attribute,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    repeat_dims = ttnn.ir.ShapeAttr.maybe_downcast(repeat_dims_attr).shape
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return input_tensor.repeat(repeats=repeat_dims).to(output_dtype)


def ttnn_where_golden(
    condition: GoldenMapTensor,
    x: GoldenMapTensor,
    y: GoldenMapTensor,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.where(condition, x, y).to(output_dtype)


def ttnn_clamp_tensor_golden(
    input_tensor: GoldenMapTensor,
    min_tensor: GoldenMapTensor,
    max_tensor: GoldenMapTensor,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.min(torch.max(input_tensor, min_tensor), max_tensor).to(output_dtype)


def ttnn_clamp_scalar_golden(
    input_tensor: GoldenMapTensor,
    min_attr: FloatAttr,
    max_attr: FloatAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    min_val = unpack_mlir_attr(min_attr)
    max_val = unpack_mlir_attr(max_attr)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.clamp(input_tensor, min=min_val, max=max_val).to(output_dtype)


def ttnn_repeat_interleave_golden(
    input_tensor: GoldenMapTensor,
    repeats_attr: IntegerAttr,
    dim_attr: IntegerAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    repeats = unpack_mlir_attr(repeats_attr)
    dim = unpack_mlir_attr(dim_attr)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.repeat_interleave(input_tensor, repeats, dim=dim).to(output_dtype)


def ttnn_leaky_relu_golden(
    input_tensor: GoldenMapTensor,
    parameter_attr: FloatAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    parameter = unpack_mlir_attr(parameter_attr)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.nn.functional.leaky_relu(input_tensor, negative_slope=parameter).to(
        output_dtype
    )


def ttnn_mish_golden(
    input_tensor: GoldenMapTensor,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.nn.functional.mish(input_tensor).to(output_dtype)


################ TTNN CCL Op Golden Functions ###############


def ttnn_all_gather_golden(
    input: GoldenMapTensor,
    all_gather_dim_attr: IntegerAttr,
    cluster_axis_attr: IntegerAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    all_gather_dim = unpack_mlir_attr(all_gather_dim_attr)
    cluster_axis = unpack_mlir_attr(cluster_axis_attr)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)

    num_devices = input.mesh_shape[0] * input.mesh_shape[1]
    if len(input.shard_map) < num_devices:
        base_tensor = input.shard_map[0]
        full_shard_map = {i: base_tensor.clone() for i in range(num_devices)}
        input = GoldenMapTensor(full_shard_map, input.mesh_shape)

    output_shards = [None] * len(input.shard_map)
    grouped_shards = input.group_by_axis(cluster_axis)
    for group in grouped_shards:
        gathered_tensor = torch.cat(list(group.values()), dim=all_gather_dim)
        for id in group.keys():
            output_shards[id] = gathered_tensor.clone().to(output_dtype)
    return GoldenMapTensor(
        {i: t for i, t in enumerate(output_shards)}, input.mesh_shape
    )


################ Debug Op Golden Functions ###############


def debug_annotate_golden(
    input_tensor: GoldenMapTensor,
    annotation_attr: StringAttr,
) -> GoldenMapTensor:
    return input_tensor.clone()


def debug_region_start_golden(
    input_tensor: GoldenMapTensor,
    region_id_attr: StringAttr,
) -> GoldenMapTensor:
    return input_tensor.clone()


def debug_region_end_golden(
    input_tensor: GoldenMapTensor,
    region_id_attr: StringAttr,
) -> GoldenMapTensor:
    return input_tensor.clone()


GOLDEN_MAPPINGS: Dict[type, Callable] = {
    # ----- TTIR OPS -----
    # Elementwise unary operations
    ttir.GetDimensionSizeOp: get_dimension_size_golden,
    ttir.AbsOp: ttir_abs_golden,
    ttir.CeilOp: torch.ceil,
    ttir.CosOp: ttir_cos_golden,
    ttir.ErfOp: ttir_erf_golden,
    ttir.ErfcOp: torch.erfc,
    ttir.FloorOp: ttir_floor_golden,
    ttir.GeluOp: torch.nn.functional.gelu,
    ttir.GeluBackwardOp: torch.ops.aten.gelu_backward,
    ttir.IsFiniteOp: ttir_isfinite_golden,
    ttir.MishOp: torch.nn.functional.mish,
    ttir.NegOp: ttir_neg_golden,
    ttir.TanOp: torch.tan,
    ttir.AtanOp: torch.atan,
    ttir.TanhOp: ttir_tanh_golden,
    ttir.ReciprocalOp: torch.reciprocal,
    ttir.ReluOp: torch.relu,
    ttir.Relu6Op: torch.nn.functional.relu6,
    ttir.RsqrtOp: ttir_rsqrt_golden,
    ttir.SigmoidOp: ttir_sigmoid_golden,
    ttir.HardsigmoidOp: ttir_hardsigmoid_golden,
    ttir.SignOp: torch.sign,
    ttir.SiluOp: silu_golden,
    ttir.SinOp: ttir_sin_golden,
    ttir.SqrtOp: ttir_sqrt_golden,
    ttir.LogOp: ttir_log_golden,
    ttir.Log1pOp: ttir_log1p_golden,
    ttir.Expm1Op: torch.expm1,
    ttir.ExpOp: ttir_exp_golden,
    # Elementwise binary operations
    ttir.AddOp: ttir_add_golden,
    ttir.Atan2Op: torch.atan2,
    ttir.MultiplyOp: ttir_multiply_golden,
    ttir.SubtractOp: ttir_subtract_golden,
    ttir.DivOp: ttir_div_golden,
    ttir.MaximumOp: ttir_maximum_golden,
    ttir.MinimumOp: ttir_minimum_golden,
    ttir.RemainderOp: torch.remainder,
    ttir.PowOp: ttir_pow_golden,
    # Comparison operations
    ttir.EqualOp: ttir_equal_golden,
    ttir.NotEqualOp: ttir_ne_golden,
    ttir.GreaterEqualOp: ttir_ge_golden,
    ttir.GreaterThanOp: ttir_greater_than_golden,
    ttir.LessEqualOp: ttir_le_golden,
    ttir.LessThanOp: ttir_lt_golden,
    # Logical operations
    ttir.LogicalAndOp: ttir_logical_and_golden,
    ttir.LogicalLeftShiftOp: logical_left_shift_golden,
    ttir.LogicalOrOp: logical_or_golden,
    ttir.LogicalRightShiftOp: ttir_logical_right_shift_golden,
    ttir.LogicalXorOp: logical_xor_golden,
    ttir.LogicalNotOp: ttir_logical_not_golden,
    # Selection operations
    ttir.WhereOp: ttir_where_golden,
    # Bitwise operations
    ttir.BitwiseAndOp: ttir_bitwise_and_golden,
    ttir.BitwiseOrOp: torch.bitwise_or,
    ttir.BitwiseXorOp: torch.bitwise_xor,
    ttir.BitwiseNotOp: torch.bitwise_not,
    # Reduction operations
    ttir.SumOp: ttir_sum_golden,
    ttir.MeanOp: mean_golden,
    ttir.MaxOp: ttir_max_golden,
    ttir.MinOp: min_golden,
    ttir.ProdOp: prod_golden,
    ttir.ReduceAndOp: ttir_reduce_and_golden,
    ttir.ReduceOrOp: ttir_reduce_or_golden,
    ttir.TopKOp: ttir_topk_golden,
    # Tensor manipulation
    ttir.SortOp: ttir_sort_golden,
    ttir.TransposeOp: transpose_golden,
    ttir.ConcatOp: ttir_concat_golden,
    ttir.RepeatOp: ttir_repeat_golden,
    ttir.RepeatInterleaveOp: repeat_interleave_golden,
    ttir.ReshapeOp: ttir_reshape_golden,
    ttir.RearrangeOp: ttir_rearrange_golden,
    ttir.SqueezeOp: squeeze_golden,
    ttir.UnsqueezeOp: unsqueeze_golden,
    ttir.ReverseOp: ttir_reverse_golden,
    ttir.PermuteOp: ttir_permute_golden,
    ttir.ClampScalarOp: clamp_scalar_golden,
    ttir.ClampTensorOp: ttir_clamp_tensor_golden,
    ttir.CumSumOp: ttir_cumsum_golden,
    ttir.BroadcastOp: ttir_broadcast_golden,
    ttir.PadOp: ttir_pad_golden,
    ttir.IndexSelectOp: select_golden,
    ttir.IndexOp: index_golden,
    ttir.SliceStaticOp: ttir_slice_golden,
    ttir.GatherOp: ttir_gather_golden,
    # Neural network operations
    ttir.SoftmaxOp: softmax_golden,
    ttir.MatmulOp: matmul_golden,
    ttir.EmbeddingOp: embedding_golden,
    ttir.EmbeddingBackwardOp: ttir_embedding_backward_golden,
    ttir.Upsample2dOp: upsample2d_golden,
    ttir.BatchNormInferenceOp: ttir_batch_norm_inference_golden,
    ttir.BatchNormTrainingOp: ttir_batch_norm_training_golden,
    ttir.LayerNormOp: ttir_layer_norm_golden,
    ttir.SplitQueryKeyValueAndSplitHeadsOp: ttir_split_query_key_value_and_split_heads_golden,
    ttir.RMSNormOp: ttir_rms_norm_golden,
    ttir.DistributedRMSNormOp: ttir_distributed_rms_norm_golden,
    # Type operations
    ttir.TypecastOp: ttir_typecast_golden,
    # Tensor creation
    ttir.ZerosOp: ttir_zeros_golden,
    ttir.OnesOp: ttir_ones_golden,
    ttir.ConstantOp: ttir_constant_golden,
    ttir.FullOp: ttir_full_golden,
    ttir.ArangeOp: ttir_arange_golden,
    ttir.RandOp: ttir_rand_golden,
    ttir.DropoutOp: ttir_dropout_golden,
    # Quantization operations
    ttir.QuantizeOp: quantize_golden,
    ttir.DequantizeOp: torch.dequantize,
    ttir.RequantizeOp: requantize_golden,
    # Complex operations
    ttir.CbrtOp: cbrt_golden,
    ttir.ConcatenateHeadsOp: ttir_concatenate_heads_golden,
    ttir.Conv2dOp: conv2d_golden,
    ttir.ConvTranspose2dOp: conv_transpose2d_golden,
    ttir.MaxPool2dOp: ttir_max_pool2d_golden,
    ttir.AvgPool2dOp: avg_pool2d_golden,
    ttir.GlobalAvgPool2dOp: global_avg_pool2d_golden,
    ttir.MaxPool2dWithIndicesOp: ttir_max_pool2d_with_indices,
    ttir.ArgMaxOp: argmax_golden,
    ttir.LinearOp: linear_golden,
    ttir.DotGeneralOp: ttir_dot_general_golden,
    ttir.ScatterOp: ttir_scatter_golden,
    # Layout operations (identity functions) — accept and ignore extra kwargs like reinterpretLayout
    ttir.ToLayoutOp: ttir_to_layout_golden,
    # Cache operations
    ttir.FillCacheOp: fill_cache_golden,
    ttir.UpdateCacheOp: update_cache_golden,
    # CCL (Collective Communication Library) operations
    ttir.MeshShardOp: ttir_mesh_shard_golden,
    ttir.AllGatherOp: ttir_all_gather_golden,
    ttir.AllReduceOp: ttir_all_reduce_golden,
    ttir.ReduceScatterOp: ttir_reduce_scatter_golden,
    ttir.CollectivePermuteOp: ttir_collective_permute_golden,
    ttir.AllToAllOp: ttir_all_to_all_golden,
    ttir.CollectiveBroadcastOp: ttir_collective_broadcast_golden,
    # Operations with parameter transformations
    ttir.LeakyReluOp: leaky_relu_golden,
    # ----- D2M OPS -----
    # D2M Layout operations (identity functions)
    d2m.ToLayoutOp: (lambda x, **kwargs: x),
    d2m.ViewLayoutOp: (lambda x, **kwargs: x),
    # ----- STABLEHLO OPS -----
    # StableHLO elementwise operations
    stablehlo.AddOp: stablehlo_add_golden,
    stablehlo.AbsOp: stablehlo_abs_golden,
    stablehlo.CeilOp: stablehlo_ceil_golden,
    stablehlo.ClampOp: stablehlo_clamp_golden,
    stablehlo.ConcatenateOp: stablehlo_concatenate_golden,
    stablehlo.CosineOp: stablehlo_cosine_golden,
    stablehlo.DivOp: stablehlo_divide_golden,
    stablehlo.ExpOp: stablehlo_exp_golden,
    stablehlo.FloorOp: stablehlo_floor_golden,
    stablehlo.ConstantOp: stablehlo_constant_golden,
    stablehlo.IotaOp: stablehlo_iota_golden,
    stablehlo.DynamicIotaOp: stablehlo_dynamic_iota_golden,
    stablehlo.BatchNormGradOp: stablehlo_batch_norm_grad_golden,
    stablehlo.BatchNormTrainingOp: stablehlo_batch_norm_training_golden,
    stablehlo.BatchNormInferenceOp: stablehlo_batch_norm_inference_golden,
    stablehlo.LogOp: stablehlo_log_golden,
    stablehlo.Log1pOp: stablehlo_log1p_golden,
    stablehlo.LogisticOp: stablehlo_logistic_golden,
    stablehlo.NegOp: stablehlo_neg_golden,
    stablehlo.ReshapeOp: stablehlo_reshape_golden,
    stablehlo.RsqrtOp: stablehlo_rsqrt_golden,
    stablehlo.SineOp: stablehlo_sine_golden,
    stablehlo.SqrtOp: stablehlo_sqrt_golden,
    stablehlo.TanOp: stablehlo_tan_golden,
    stablehlo.TanhOp: stablehlo_tanh_golden,
    stablehlo.AndOp: stablehlo_and_golden,
    stablehlo.OrOp: stablehlo_or_golden,
    stablehlo.XorOp: stablehlo_xor_golden,
    stablehlo.NotOp: stablehlo_not_golden,
    stablehlo.SliceOp: stablehlo_slice_golden,
    stablehlo.GetDimensionSizeOp: stablehlo_get_dimension_size_golden,
    stablehlo.MaxOp: stablehlo_maximum_golden,
    stablehlo.MinOp: stablehlo_minimum_golden,
    stablehlo.MulOp: stablehlo_multiply_golden,
    # bitcast conversion operation
    stablehlo.BroadcastInDimOp: torch.broadcast_to,
    stablehlo.SubtractOp: stablehlo_subtract_golden,
    stablehlo.PowOp: stablehlo_pow_golden,
    stablehlo.ShiftRightLogicalOp: stablehlo_shift_right_logical_golden,
    stablehlo.ReverseOp: stablehlo_reverse_golden,
    stablehlo.DotGeneralOp: dot_general_golden,
    stablehlo.DynamicSliceOp: dynamic_slice_golden,
    stablehlo.DynamicUpdateSliceOp: stablehlo_dynamic_update_slice_golden,
    stablehlo.ConvolutionOp: stablehlo_convolution_golden,
    stablehlo.SortOp: stablehlo_sort_golden,
    # StableHLO tensor manipulation operations
    stablehlo.TransposeOp: stablehlo_transpose_golden,
    stablehlo.SelectOp: stablehlo_select_golden,
    stablehlo.PadOp: stablehlo_pad_golden,
    # CCL (Collective Communication Library) operations
    stablehlo.AllGatherOp: stablehlo_all_gather_golden,
    stablehlo.AllReduceOp: stablehlo_all_reduce_golden,
    stablehlo.ReduceScatterOp: stablehlo_reduce_scatter_golden,
    stablehlo.ReduceWindowOp: stablehlo_reduce_window_golden,
    stablehlo.CollectivePermuteOp: stablehlo_collective_permute_golden,
    stablehlo.AllToAllOp: stablehlo_all_to_all_golden,
    stablehlo.CollectiveBroadcastOp: stablehlo_collective_broadcast_golden,
    # ----- SDY OPS -----
    sdy.ShardingConstraintOp: sdy_sharding_constraint_golden,
    sdy.ReshardOp: sdy_reshard_golden,
    sdy.AllGatherOp: sdy_all_gather_golden,
    # ----- TTNN OPS -----
    # Elementwise unary operations
    ttnn.AbsOp: ttnn_abs_golden,
    ttnn.CbrtOp: ttnn_cbrt_golden,
    ttnn.CeilOp: ttnn_ceil_golden,
    ttnn.CosOp: ttnn_cos_golden,
    ttnn.ErfOp: ttnn_erf_golden,
    ttnn.ErfcOp: ttnn_erfc_golden,
    ttnn.FloorOp: ttnn_floor_golden,
    ttnn.GeluOp: ttnn_gelu_golden,
    ttnn.IsFiniteOp: ttnn_isfinite_golden,
    ttnn.MishOp: ttnn_mish_golden,
    ttnn.NegOp: ttnn_neg_golden,
    ttnn.TanOp: ttnn_tan_golden,
    ttnn.AtanOp: ttnn_atan_golden,
    ttnn.TanhOp: ttnn_tanh_golden,
    ttnn.ReciprocalOp: ttnn_reciprocal_golden,
    ttnn.ReluOp: ttnn_relu_golden,
    ttnn.Relu6Op: ttnn_relu6_golden,
    ttnn.RsqrtOp: ttnn_rsqrt_golden,
    ttnn.SigmoidOp: ttnn_sigmoid_golden,
    ttnn.SignOp: ttnn_sign_golden,
    ttnn.SiluOp: ttnn_silu_golden,
    ttnn.SinOp: ttnn_sin_golden,
    ttnn.SqrtOp: ttnn_sqrt_golden,
    ttnn.LogOp: ttnn_log_golden,
    ttnn.Log1pOp: ttnn_log1p_golden,
    ttnn.Expm1Op: ttnn_expm1_golden,
    ttnn.ExpOp: ttnn_exp_golden,
    ttnn.LeakyReluOp: ttnn_leaky_relu_golden,
    # Elementwise binary operations
    ttnn.AddOp: ttnn_add_golden,
    ttnn.Atan2Op: ttnn_atan2_golden,
    ttnn.MultiplyOp: ttnn_multiply_golden,
    ttnn.SubtractOp: ttnn_subtract_golden,
    ttnn.DivideOp: ttnn_divide_golden,
    ttnn.MaximumOp: ttnn_maximum_golden,
    ttnn.MinimumOp: ttnn_minimum_golden,
    ttnn.RemainderOp: ttnn_remainder_golden,
    ttnn.PowTensorOp: ttnn_pow_tensor_golden,
    # Comparison operations
    ttnn.EqualOp: ttnn_eq_golden,
    ttnn.NotEqualOp: ttnn_ne_golden,
    ttnn.GreaterEqualOp: ttnn_ge_golden,
    ttnn.GreaterThanOp: ttnn_gt_golden,
    ttnn.LessEqualOp: ttnn_le_golden,
    ttnn.LessThanOp: ttnn_lt_golden,
    # Logical operations
    ttnn.LogicalAndOp: ttnn_logical_and_golden,
    ttnn.LogicalLeftShiftOp: ttnn_logical_left_shift_golden,
    ttnn.LogicalOrOp: ttnn_logical_or_golden,
    ttnn.LogicalRightShiftOp: ttnn_logical_right_shift_golden,
    ttnn.LogicalXorOp: ttnn_logical_xor_golden,
    ttnn.LogicalNotOp: ttnn_logical_not_golden,
    # Selection operations
    ttnn.WhereOp: ttnn_where_golden,
    # Type operations
    ttnn.TypecastOp: ttnn_typecast_golden,
    # Bitwise operations
    ttnn.BitwiseAndOp: ttnn_bitwise_and_golden,
    ttnn.BitwiseOrOp: ttnn_bitwise_or_golden,
    ttnn.BitwiseXorOp: ttnn_bitwise_xor_golden,
    ttnn.BitwiseNotOp: ttnn_bitwise_not_golden,
    # Complex operations
    ttnn.MatmulOp: ttnn_matmul_golden,
    ttnn.LinearOp: ttnn_linear_golden,
    ttnn.LayerNormOp: ttnn_layer_norm_golden,
    ttnn.RMSNormOp: rms_norm_golden,
    # Tensor manipulation
    ttnn.ConcatOp: ttnn_concat_golden,
    ttnn.RepeatOp: ttnn_repeat_golden,
    ttnn.RepeatInterleaveOp: ttnn_repeat_interleave_golden,
    ttnn.ClampScalarOp: ttnn_clamp_scalar_golden,
    ttnn.ClampTensorOp: ttnn_clamp_tensor_golden,
    # CCL (Collective Communication Library) operations
    ttnn.AllGatherOp: ttnn_all_gather_golden,
    # ----- DEBUG OPS -----
    debug.AnnotateOp: debug_annotate_golden,
    debug.RegionStartOp: debug_region_start_golden,
    debug.RegionEndOp: debug_region_end_golden,
}


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

    assert False, f"No golden function found for TTIR operation: {ttir_op_class}"
