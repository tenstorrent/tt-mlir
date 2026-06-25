# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Golden function mappings for TTIR, StableHLO, TTNN, and related dialects.

Two registries live here, split by calling convention rather than by dialect:

- ``GOLDEN_MAPPINGS`` — keyed by op class across all supported dialects
  (TTIR, StableHLO, TTNN, D2M, SDY, Debug). Goldens take torch values plus raw
  MLIR attributes; consumed by the builders in ``tools/builder/`` while
  constructing IR.
- ``CHISEL_GOLDEN_MAPPINGS`` — keyed by TTNN op class. Goldens take
  ``(op, inputs: Dict[str, GoldenMapTensor])`` and return one tensor per SSA
  result followed by one per provided in-place operand; consumed by
  ``tools/chisel/chisel/executor.py`` when replaying flatbuffers.

Each golden function is a PyTorch reference implementation that produces the
expected output for comparison with the corresponding op's result.
"""

from __future__ import annotations
from typing import Dict, Callable, Any, Optional, Union, List, Tuple, Iterable, Iterator
import itertools
import operator
import re
import einops
import torch
import torch.nn.functional
from ttmlir.dialects import ttir, stablehlo, d2m, ttnn, ttcore, sdy, debug, func
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
        "bool",
    ]
    _MUTATING_METHODS = {
        name: (
            lambda name: lambda shard, *args, **kwargs: getattr(shard, name)(
                *args, **kwargs
            )
        )(name)
        for name in _MUTATING_METHOD_NAMES
    }

    # Torch matmul functions that are too slow on CPUs w/o hardware bf16 support.
    _BF16_UPCAST_MM_FUNCS = frozenset({torch.matmul, torch.mm, torch.bmm, torch.einsum})

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

    def __str__(self) -> str:
        return (
            f"GoldenMapTensor(mesh_shape={self._mesh_shape}, shards={self._shard_map})"
        )

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

        # Transparently upcast bf16 matmul to f32 and cast back so it uses the fast BLAS backend.
        if func in cls._BF16_UPCAST_MM_FUNCS and any(
            st.dtype == torch.bfloat16 for st in st_inputs
        ):
            _orig_func = func
            # Torch promotes mixed-precision matmuls (bf16 @ f32 -> f32), so we only downcast if there's no f32.
            has_f32_input = any(st.dtype == torch.float32 for st in st_inputs)

            print(f"Upcasting bf16 matmul to f32 for {_orig_func.__name__}")

            def func(*a, **kw):
                a = tuple(
                    (
                        x.float()
                        if isinstance(x, torch.Tensor) and x.dtype == torch.bfloat16
                        else x
                    )
                    for x in a
                )
                result = _orig_func(*a, **kw)
                if not has_f32_input:
                    return result.to(torch.bfloat16)
                return result

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


def _attr_get(attrs, key, default=None):
    """Safe attribute access for MLIR OpAttributeMap (which lacks .get())."""
    return attrs[key] if key in attrs else default


def _attr_get_value(attrs, key, default=None):
    """Return `unpack_mlir_attr(attrs[key])` or `default` if `key` is absent."""
    return unpack_mlir_attr(attrs[key]) if key in attrs else default


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


def conv3d_golden(
    input_tensor: GoldenMapTensor,
    weight: GoldenMapTensor,
    bias: Optional[GoldenMapTensor],
    stride: Union[IntegerAttr, DenseI32ArrayAttr],
    padding: Union[IntegerAttr, DenseI32ArrayAttr],
    groups: IntegerAttr,
    batch_dim: IntegerAttr,
    depth_dim: IntegerAttr,
    height_dim: IntegerAttr,
    width_dim: IntegerAttr,
    channel_dim: IntegerAttr,
    padding_mode: StringAttr,
) -> GoldenMapTensor:
    """
    Custom golden function for conv3d with layout transformation.

    Parameters
    ----------
    input_tensor : GoldenMapTensor
        Input tensor in (N, D, H, W, C) format
    weight : GoldenMapTensor
        Weight tensor in (C_out, C_in, K_D, K_H, K_W) format
    bias : Optional[GoldenMapTensor]
        Optional bias tensor in (1, 1, 1, 1, C_out) format
    stride : Union[IntegerAttr, DenseI32ArrayAttr]
        Stride for depth, height, width
    padding : Union[IntegerAttr, DenseI32ArrayAttr]
        Padding for depth, height, width (symmetric)
    groups : IntegerAttr
        Number of groups for grouped convolution
    batch_dim : IntegerAttr
        Batch dimension index
    depth_dim : IntegerAttr
        Depth dimension index
    height_dim : IntegerAttr
        Height dimension index
    width_dim : IntegerAttr
        Width dimension index
    channel_dim : IntegerAttr
        Channel dimension index
    padding_mode : StringAttr
        Padding mode ("zeros" or "replicate")

    Returns
    -------
    GoldenMapTensor
        Result of 3D convolution with layout transformation
    """
    if bias is not None:
        bias = bias.squeeze()

    stride = unpack_mlir_attr(stride)
    padding = unpack_mlir_attr(padding)
    groups = unpack_mlir_attr(groups)
    padding_mode_str = unpack_mlir_attr(padding_mode)

    batch_dim = unpack_mlir_attr(batch_dim)
    depth_dim = unpack_mlir_attr(depth_dim)
    height_dim = unpack_mlir_attr(height_dim)
    width_dim = unpack_mlir_attr(width_dim)
    channel_dim = unpack_mlir_attr(channel_dim)

    # Compute permutation to convert any layout to NCDHW
    to_ncdhw_perm = [batch_dim, channel_dim, depth_dim, height_dim, width_dim]

    is_ncdhw = to_ncdhw_perm == [0, 1, 2, 3, 4]

    copied_input_tensor = input_tensor.clone()
    if not is_ncdhw:
        copied_input_tensor = copied_input_tensor.permute(to_ncdhw_perm)

    result = torch.nn.functional.conv3d(
        copied_input_tensor,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=1,
        groups=groups,
    )

    if not is_ncdhw:
        from_ncdhw_perm = [0] * 5
        for i, p in enumerate(to_ncdhw_perm):
            from_ncdhw_perm[p] = i
        result = result.permute(from_ncdhw_perm)
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


def ttir_distributed_layer_norm_golden(
    input: GoldenMapTensor,
    weight: Optional[GoldenMapTensor],
    bias: Optional[GoldenMapTensor],
    residual: Optional[GoldenMapTensor],
    cluster_axis_attr: IntegerAttr,
    epsilon_attr: FloatAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    """Distributed layer normalization golden.

    Simulates the distributed layer_norm_pre_all_gather + all_gather +
    layer_norm_post_all_gather pipeline: for each group of devices along
    ``cluster_axis``, concatenate the per-device shards to compute
    globally-correct mean and variance statistics, apply layer norm +
    optional weight/bias on the full tensor, then chunk the result back
    so each device gets only its local portion. The per-device output
    shape equals the per-device input shape (only the stats are
    all-gathered, not the data).

    Parameters
    ----------
    input : GoldenMapTensor
        Per-device input tensor shards.
    weight : Optional[GoldenMapTensor]
        Per-device weight (gamma) shards. If present, applied after normalization.
    bias : Optional[GoldenMapTensor]
        Per-device bias (beta) shards. If present, added after weight scaling.
    residual : Optional[GoldenMapTensor]
        Per-device residual shards. If present, added to input before
        normalization.
    cluster_axis_attr : IntegerAttr
        Mesh axis (0 or 1) along which devices exchange layer norm statistics.
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

        # Concatenate shards along last dim to get full tensor for global
        # layer norm statistics (mean and variance)
        full_tensor = torch.cat(group_tensors, dim=-1).float()

        # Compute layer norm on full tensor
        normalized_shape = [full_tensor.shape[-1]]
        weight_tensor = None
        if weight is not None:
            weight_shards = [weight.shard_map[id] for id in group_ids]
            weight_tensor = torch.cat(weight_shards, dim=-1)

        bias_tensor = None
        if bias is not None:
            bias_shards = [bias.shard_map[id] for id in group_ids]
            bias_tensor = torch.cat(bias_shards, dim=-1)

        ln_result = torch.nn.functional.layer_norm(
            full_tensor,
            normalized_shape=normalized_shape,
            weight=weight_tensor.float() if weight_tensor is not None else None,
            bias=bias_tensor.float() if bias_tensor is not None else None,
            eps=epsilon,
        )

        ln_result = ln_result.to(output_dtype)

        # Split back into per-device chunks (output shape == input shape).
        per_shard_chunks = torch.chunk(ln_result, len(group_ids), dim=-1)
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
    weight_float = weight.float() if weight is not None else None
    bias_float = bias.float() if bias is not None else None

    return torch.nn.functional.layer_norm(
        input_float,
        normalized_shape=normalized_shape,
        weight=weight_float,
        bias=bias_float,
        eps=epsilon,
    ).to(output_dtype)


def ttir_group_norm_golden(
    input: GoldenMapTensor,
    weight: Optional[GoldenMapTensor],
    bias: Optional[GoldenMapTensor],
    num_groups,
    epsilon: FloatAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    num_groups = unpack_mlir_attr(num_groups)
    epsilon = unpack_mlir_attr(epsilon)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    input_float = input.float()

    # torch.group_norm expects [N, C, ...] (channels at dim=1), but the TTIR
    # GroupNorm op uses channels-last: [N, 1, H*W, C]. Permute to NCHW,
    # compute, then permute back.
    if input_float.dim() == 4 and input_float.shape[1] == 1:
        input_nchw = input_float.permute(0, 3, 1, 2)
        result = torch.nn.functional.group_norm(
            input_nchw,
            num_groups=num_groups,
            weight=weight,
            bias=bias,
            eps=epsilon,
        )
        return result.permute(0, 2, 3, 1).to(output_dtype)

    return torch.nn.functional.group_norm(
        input_float,
        num_groups=num_groups,
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


def ttir_sparse_matmul_golden(
    a: GoldenMapTensor,
    b: GoldenMapTensor,
    sparsity: GoldenMapTensor,
    is_input_a_sparse_attr,
    is_input_b_sparse_attr,
    nnz_attr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    """Golden function for sparse_matmul. On CPU, performs dense matmul (sparsity
    is applied at runtime on device)."""
    # Unpack MLIR attributes
    is_input_a_sparse = unpack_mlir_attr(is_input_a_sparse_attr)
    is_input_b_sparse = unpack_mlir_attr(is_input_b_sparse_attr)
    nnz = unpack_mlir_attr(nnz_attr) if nnz_attr is not None else None

    # For golden: ignore sparsity mask, compute dense batched matmul.
    # a: [..., M, K], b: [1, E, K, N] -> output depends on mode.
    if is_input_b_sparse and not is_input_a_sparse:
        # Column-parallel: a [A, B, M, K], b [1, E, K, N] -> [A, B, 1, E, M, N]
        # Use einsum: a(abmk) x b(xekn) -> (abxemn) where x=1
        result = torch.einsum("abmk,xekn->abxemn", a, b)
        return result
    elif is_input_a_sparse and not is_input_b_sparse:
        # Row-parallel: a [A, E, M, K], b [1, E, K, N] -> [A, E, M, N]
        return torch.matmul(a, b)
    else:
        # Both sparse: a [1, E, M, K], b [1, E, K, N] -> [1, E, M, N]
        return torch.matmul(a, b)


def ttir_all_to_all_dispatch_golden(
    input_tensor: GoldenMapTensor,
    expert_indices: GoldenMapTensor,
    expert_mapping: GoldenMapTensor,
    num_devices_attr,
    cluster_axis_attr,
    dispatched_type_mlir: Type,
    metadata_type_mlir: Type,
) -> Tuple[GoldenMapTensor, GoldenMapTensor]:
    """Golden for dispatch with layout-aware token expansion."""
    # Unpack MLIR attributes
    num_devices = unpack_mlir_attr(num_devices_attr)
    cluster_axis = unpack_mlir_attr(cluster_axis_attr)

    D = num_devices if isinstance(num_devices, int) else 2

    def _to_dispatch_layout(tensor):
        # Support both [B, 1, S, C] and [B, S, 1, C].
        if tensor.shape[1] == 1:
            return tensor.permute(1, 0, 2, 3)
        if tensor.shape[2] == 1:
            return tensor.permute(2, 0, 1, 3)
        return tensor.unsqueeze(0)

    dispatched = _to_dispatch_layout(input_tensor).repeat(1, D, 1, 1)
    metadata = _to_dispatch_layout(expert_indices).repeat(1, D, 1, 1)
    return dispatched, metadata


def ttir_all_to_all_dispatch_metadata_golden(
    input_tensor: GoldenMapTensor,
    expert_indices: GoldenMapTensor,
    expert_scores: GoldenMapTensor,
    expert_mapping: GoldenMapTensor,
    num_devices_attr,
    cluster_axis_attr,
    dispatched_type_mlir: Type,
    indices_type_mlir: Type,
    scores_type_mlir: Type,
) -> Tuple:
    """Cross-shard golden for all_to_all_dispatch_metadata.

    Mirrors tt-metal's gen_tensors_for_metadata_op / get_output_tensor:
    - Dispatched: sparse routing — for each token and each of its K selected
      experts, the token is placed on the device that owns that expert.
      Non-routed slots are filled with zeros.
    - Indices (metadata): all-gathered — every ring device gets the full set
      of expert indices from all ring devices.
    - Scores: all-gathered — same as indices.

    expert_mapping has new format [1, 1, D, E] where entry [0, 0, d, e] is
    the linearized device ID that owns expert e (same for all d).
    """
    # Unpack MLIR attributes
    num_devices = unpack_mlir_attr(num_devices_attr)
    cluster_axis = unpack_mlir_attr(cluster_axis_attr)

    num_devs = num_devices if isinstance(num_devices, int) else 2
    mesh_shape = input_tensor.mesh_shape
    grouped_inputs = input_tensor.group_by_axis(cluster_axis)
    grouped_indices = expert_indices.group_by_axis(cluster_axis)
    grouped_scores = expert_scores.group_by_axis(cluster_axis)

    # expert_mapping is replicated — get from any device
    mapping_tensor = list(expert_mapping._shard_map.values())[0]
    # Shape is [1, 1, D, E] — squeeze to [D, E], use row 0 since all rows identical
    mapping_2d = mapping_tensor.reshape(-1, mapping_tensor.shape[-1])  # [D, E]
    mapping_row = mapping_2d[0]  # [E] — mapping_row[e] = device_id owning expert e

    out_dispatched = {}
    out_indices = {}
    out_scores = {}

    for ring_group_inp, ring_group_idx, ring_group_scr in zip(
        grouped_inputs, grouped_indices, grouped_scores
    ):
        ring_device_ids = sorted(ring_group_inp.keys())
        sample = ring_group_inp[ring_device_ids[0]]
        M = sample.reshape(-1, sample.shape[-1]).shape[0]
        H = sample.shape[-1]
        K = ring_group_idx[ring_device_ids[0]].shape[-1]
        total_tokens = num_devs * M

        # Reconstruct full tensors across the ring in ring-position order
        full_input = torch.cat(
            [ring_group_inp[d].reshape(-1, H) for d in ring_device_ids], dim=0
        )  # [total_tokens, H]
        full_idx = torch.cat(
            [ring_group_idx[d].reshape(-1, K) for d in ring_device_ids], dim=0
        )  # [total_tokens, K]
        full_scr = torch.cat(
            [ring_group_scr[d].reshape(-1, K) for d in ring_device_ids], dim=0
        )  # [total_tokens, K]

        # --- Dispatched: sparse expert-based routing ---
        # Initialize with zeros so non-routed slots are identifiable.
        disp_per_dev = {
            d: torch.zeros(1, total_tokens, H, dtype=full_input.dtype)
            for d in ring_device_ids
        }

        for t in range(total_tokens):
            token_data = full_input[t]  # [H]
            for k in range(K):
                expert_id = int(full_idx[t, k].item())
                target_device_id = int(mapping_row[expert_id].item())
                # Only route if target device is in this ring
                if target_device_id in disp_per_dev:
                    disp_per_dev[target_device_id][0, t, :] = token_data

        # --- Indices/Scores: all-gathered (every device gets full set) ---
        # 3D shapes matching metal kernel output: [1, tokens_global, C]
        idx_out = full_idx.reshape(1, total_tokens, K)
        scr_out = full_scr.reshape(1, total_tokens, K)

        for dev_id in ring_device_ids:
            out_dispatched[dev_id] = disp_per_dev[
                dev_id
            ]  # already [1, total_tokens, H]
            out_indices[dev_id] = idx_out.clone()
            out_scores[dev_id] = scr_out.clone()

    return (
        GoldenMapTensor(out_dispatched, mesh_shape),
        GoldenMapTensor(out_indices, mesh_shape),
        GoldenMapTensor(out_scores, mesh_shape),
    )


def _swiglu_reference(gate, up, alpha=1.702, clamp_limit=7.0):
    """SwiGLU activation matching tt-metal's SFPU implementation."""
    gate_c = torch.clamp(gate, max=clamp_limit)
    up_c = torch.clamp(up, min=-clamp_limit, max=clamp_limit)
    return (up_c + 1.0) * gate_c * torch.sigmoid(alpha * gate_c)


def moe_gpt_golden(
    input_tensor: GoldenMapTensor,
    expert_indices: GoldenMapTensor,
    expert_scores: GoldenMapTensor,
    expert_mapping: GoldenMapTensor,
    raw_w0: torch.Tensor,
    raw_w1: torch.Tensor,
    raw_w2: torch.Tensor,
    hidden_size_attr,
    cluster_axis_attr,
    num_worker_cores_attr,
    token_counts_type_mlir: Type,
    activation_records_type_mlir: Type,
    token_indices_type_mlir: Type,
    tilize_out_type_mlir: Type,
    tilize_out_rm_type_mlir: Type,
    tilize_out_shape: Tuple = None,
) -> Tuple:
    """Cross-shard golden for moe_gpt.

    Ports tt-metal's test_moe_gpt_e2e.py reference implementation:
    - Outputs 0-2: routing metadata (token_counts, activation_records,
      token_indices) computed from indices/scores/mapping.
    - Outputs 3-4: MLP compute (W0/W1 -> SwiGLU -> W2) using raw weights.

    raw_w0, raw_w1, raw_w2 are the unprepared weights [L, E_per_device, K, N]
    (before interleave/shard/pad). These must be provided by the test since the
    op inputs are the prepared tensors which are hard to unpack.
    """
    # Unpack MLIR attributes
    hidden_size = unpack_mlir_attr(hidden_size_attr)
    cluster_axis = unpack_mlir_attr(cluster_axis_attr)
    num_worker_cores = unpack_mlir_attr(num_worker_cores_attr)

    L1_ALIGN = 16  # l1_alignment on Wormhole

    mesh_shape = input_tensor.mesh_shape
    # expert_mapping is replicated — get from any device
    mapping_tensor = list(expert_mapping._shard_map.values())[0]
    mapping_2d = mapping_tensor.reshape(-1, mapping_tensor.shape[-1])
    mapping_row = mapping_2d[0]  # [E_total]
    E_total = mapping_row.shape[0]

    # Group inputs by cluster axis (ring groups)
    grouped_input = input_tensor.group_by_axis(cluster_axis)
    grouped_indices = expert_indices.group_by_axis(cluster_axis)
    grouped_scores = expert_scores.group_by_axis(cluster_axis)

    out_tc = {}
    out_act = {}
    out_et = {}
    out_tile = {}
    out_tile_rm = {}

    L = raw_w0.shape[0]  # layers (1)
    E_per_device = raw_w0.shape[1]
    K = hidden_size
    N = raw_w0.shape[3]  # intermediate size

    for ring_group_inp, ring_group_idx, ring_group_scr in zip(
        grouped_input, grouped_indices, grouped_scores
    ):
        ring_device_ids = sorted(ring_group_inp.keys())
        ring_devices = len(ring_device_ids)

        # input_tensor per device: [total_tokens, H] (sparse buffer from dispatch)
        sample = ring_group_inp[ring_device_ids[0]]
        total_tokens = sample.reshape(-1, K).shape[0]
        M = total_tokens // ring_devices  # tokens per device

        # expert_indices/scores per device: [1, total_tokens, K_sel] (all-gathered)
        K_sel = (
            ring_group_idx[ring_device_ids[0]]
            .reshape(-1, ring_group_idx[ring_device_ids[0]].shape[-1])
            .shape[-1]
        )

        # Get full all-gathered indices/scores (same on all ring devices)
        full_idx = ring_group_idx[ring_device_ids[0]].reshape(total_tokens, K_sel)
        full_scr = ring_group_scr[ring_device_ids[0]].reshape(total_tokens, K_sel)

        for ring_pos, dev_id in enumerate(ring_device_ids):
            # Determine local expert global IDs for this device
            local_expert_global_ids = []
            for e in range(E_total):
                if int(mapping_row[e].item()) == dev_id:
                    local_expert_global_ids.append(e)
            local_expert_global_ids = sorted(local_expert_global_ids)[:E_per_device]

            # --- Output 0: token_counts ---
            tc_elements = (E_per_device * 4 + L1_ALIGN - 1) // L1_ALIGN * L1_ALIGN // 4
            token_counts = torch.zeros(1, tc_elements, dtype=torch.int32)

            # --- Build per-expert token lists and routing metadata ---
            # activation_records row: [token_id, k_idx_0..E-1, score_0..E-1, pad]
            act_row_bytes = (2 * E_per_device + 1) * 4
            act_row_bytes_aligned = (
                (act_row_bytes + L1_ALIGN - 1) // L1_ALIGN * L1_ALIGN
            )
            act_row_stride = act_row_bytes_aligned // 4
            act_total_elements = (total_tokens + 1) * act_row_stride
            activation_records = torch.zeros(1, act_total_elements, dtype=torch.int32)

            # e_t buffer
            e_t_entry_size = (4 + L1_ALIGN - 1) // L1_ALIGN * L1_ALIGN // 4  # = 4
            e_t_row_elements = (total_tokens + 1) * e_t_entry_size
            token_indices = torch.zeros(
                E_per_device, e_t_row_elements, dtype=torch.int32
            )

            counts = [0] * E_per_device
            act_row_idx = 0

            for t in range(total_tokens):
                activated_for_any = False
                # Check each local expert
                row_data = torch.full((2 * E_per_device + 1,), 0, dtype=torch.int32)
                row_data[0] = t  # token_id
                for local_e_idx in range(E_per_device):
                    row_data[1 + local_e_idx] = K_sel + 1  # sentinel: not selected

                for local_e_idx, global_e in enumerate(local_expert_global_ids):
                    for k in range(K_sel):
                        if int(full_idx[t, k].item()) == global_e:
                            row_data[1 + local_e_idx] = k
                            # Score as uint32 bits
                            score_bf16 = full_scr[t, k].to(torch.bfloat16)
                            # bf16 → raw uint16 bits (numpy doesn't support bf16)
                            score_bits = int.from_bytes(
                                score_bf16.view(torch.int16).numpy().tobytes()[:2],
                                "little",
                            )
                            row_data[1 + E_per_device + local_e_idx] = score_bits

                            # Add to e_t buffer
                            et_offset = (
                                local_e_idx * e_t_row_elements
                                + counts[local_e_idx] * e_t_entry_size
                            )
                            token_indices[
                                local_e_idx, counts[local_e_idx] * e_t_entry_size
                            ] = t

                            counts[local_e_idx] += 1
                            activated_for_any = True
                            break  # each expert matches at most one k-slot

                if activated_for_any:
                    offset = act_row_idx * act_row_stride
                    for j in range(2 * E_per_device + 1):
                        activation_records[0, offset + j] = row_data[j]
                    act_row_idx += 1

            # Sentinel row
            sentinel_offset = act_row_idx * act_row_stride
            if sentinel_offset < act_total_elements:
                activation_records[0, sentinel_offset] = -1  # 0xFFFFFFFF

            # Write counts
            for e in range(E_per_device):
                token_counts[0, e] = counts[e]

            # Sentinel in e_t buffer
            for e in range(E_per_device):
                sentinel_pos = counts[e] * e_t_entry_size
                if sentinel_pos < e_t_row_elements:
                    token_indices[e, sentinel_pos] = -1

            out_tc[dev_id] = token_counts
            out_act[dev_id] = activation_records
            out_et[dev_id] = token_indices

            # --- Outputs 3-4: MLP compute (W0/W1 -> SwiGLU -> W2) ---
            # The kernel writes MLP results into an L1 HEIGHT_SHARDED buffer
            # shape (num_worker_cores, 2, 32, K) where only the combine cores
            # (first height_shard_dim * width_shard_dim slots, flat-indexed by
            # dhs * width_shard_dim + dws) hold meaningful data; the remaining
            # worker-core slots are uninitialized. We replicate the combine
            # kernel's write pattern from dm1.cpp/moe_gpt_program_factory.cpp:
            #   - Tokens for each expert are distributed across height_shard_dim
            #     groups using floor+remainder.
            #   - Each expert occupies 32 rows within a combine core's
            #     [2, 32] = 64-row shard (so up to 2 experts fit per core
            #     width-shard; with E_per_device=4 two rows layer into the
            #     dim-1 axis).
            #   - Each token's H-dimension values are split into width_shard_dim
            #     column bands placed across width_shard_dim combine cores.
            sparse_input = ring_group_inp[dev_id].reshape(total_tokens, K)

            height_shard_dim = 4
            width_shard_dim = 3
            TILE = 32
            combine_shard_width_tiles = K // TILE // width_shard_dim

            tile_shape = tilize_out_shape or (num_worker_cores, 2, TILE, K)
            tile_golden = torch.zeros(tile_shape, dtype=torch.bfloat16)

            for local_e_idx, global_e in enumerate(local_expert_global_ids):
                tokens_for_expert = []
                for t in range(total_tokens):
                    for k_slot in range(K_sel):
                        if int(full_idx[t, k_slot].item()) == global_e:
                            tokens_for_expert.append(sparse_input[t, :])
                            break

                if not tokens_for_expert:
                    continue

                x = torch.stack(tokens_for_expert, dim=0).float()
                with torch.no_grad():
                    w0 = raw_w0[0, local_e_idx].float()
                    w1 = raw_w1[0, local_e_idx].float()
                    w2 = raw_w2[0, local_e_idx].float()
                    gate = x @ w0
                    up = x @ w1
                    activated = _swiglu_reference(gate, up)
                    mlp = (activated @ w2).to(torch.bfloat16)  # [active, K]

                active = mlp.shape[0]
                tps = active // height_shard_dim
                rem = active % height_shard_dim

                dhs, srow = 0, 0
                for bt in range(active):
                    cap = tps + (1 if dhs < rem else 0)
                    if cap == 0:
                        break

                    # Per-expert 32-row region. The [2, 32] axis only
                    # accommodates 2 such regions per combine core (64 rows
                    # total). For E_per_device > 2 the remaining experts
                    # land in some layout we don't yet model, so skip them.
                    dst_row_flat = local_e_idx * TILE + srow
                    dim1 = dst_row_flat // TILE
                    dim2 = dst_row_flat % TILE
                    if dim1 >= tile_shape[1]:
                        srow += 1
                        if srow == cap:
                            dhs += 1
                            srow = 0
                        continue

                    for dws in range(width_shard_dim):
                        cc = dhs * width_shard_dim + dws
                        col_lo = dws * combine_shard_width_tiles * TILE
                        col_hi = col_lo + combine_shard_width_tiles * TILE
                        tile_golden[cc, dim1, dim2, col_lo:col_hi] = mlp[
                            bt, col_lo:col_hi
                        ]

                    srow += 1
                    if srow == cap:
                        dhs += 1
                        srow = 0

            out_tile[dev_id] = tile_golden
            out_tile_rm[dev_id] = tile_golden.clone()

    return (
        GoldenMapTensor(out_tc, mesh_shape),
        GoldenMapTensor(out_act, mesh_shape),
        GoldenMapTensor(out_et, mesh_shape),
        GoldenMapTensor(out_tile, mesh_shape),
        GoldenMapTensor(out_tile_rm, mesh_shape),
    )


def ttir_all_to_all_combine_golden(
    input_tensor: GoldenMapTensor,
    expert_metadata: GoldenMapTensor,
    expert_mapping: GoldenMapTensor,
    num_devices_attr,
    cluster_axis_attr,
    num_experts_per_tok_attr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    """
    Metadata-aware golden for combine.

    The combine output is routed by `expert_metadata` slots and ownership in
    `expert_mapping`. If a mapping is invalid/ambiguous, unmatched slots remain zero.
    """
    # Unpack MLIR attributes
    num_devices = unpack_mlir_attr(num_devices_attr)
    cluster_axis = unpack_mlir_attr(cluster_axis_attr)
    num_experts_per_tok = unpack_mlir_attr(num_experts_per_tok_attr)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)

    if not isinstance(input_tensor, GoldenMapTensor):
        D = num_devices if isinstance(num_devices, int) else 2
        B = max(1, input_tensor.shape[1] // D)
        return input_tensor[:, :B, :, :].to(output_dtype)

    grouped_inputs = input_tensor.group_by_axis(cluster_axis)
    grouped_metadata = expert_metadata.group_by_axis(cluster_axis)

    mapping_ref = next(iter(expert_mapping.shard_map.values()))
    num_experts = int(mapping_ref.shape[2])
    num_mapping_devices = int(mapping_ref.shape[3])

    output_shards: Dict[int, torch.Tensor] = {}
    for group_idx, group_inputs in enumerate(grouped_inputs):
        group_ids = list(group_inputs.keys())
        if len(group_ids) == 0:
            continue

        group_size = len(group_ids)
        group_metadata = grouped_metadata[group_idx]
        metadata_ref = next(iter(group_metadata.values()))
        batch_global = int(metadata_ref.shape[1])
        seq_global = int(metadata_ref.shape[2])
        k_slots = int(metadata_ref.shape[3])
        hidden_size = int(next(iter(group_inputs.values())).shape[-1])

        local_batch = max(1, batch_global // max(1, group_size))
        local_seq = seq_global

        for device_id, shard in group_inputs.items():
            output_shards[device_id] = torch.zeros(
                (k_slots, local_batch, local_seq, hidden_size),
                dtype=shard.dtype,
                device=shard.device,
            )

        local_experts_by_device: Dict[int, List[int]] = {}
        for src_id in group_ids:
            mapping_device_idx = (
                src_id if src_id < num_mapping_devices else src_id % num_mapping_devices
            )
            local_experts: List[int] = []
            for expert_idx in range(num_experts):
                if int(mapping_ref[0, 0, expert_idx, mapping_device_idx].item()) == 1:
                    local_experts.append(expert_idx)
            local_experts_by_device[src_id] = local_experts

        experts_in_metadata = set()
        for md_shard in group_metadata.values():
            for value in md_shard[0, :, :, :k_slots].reshape(-1):
                expert_idx = int(value.item())
                if 0 <= expert_idx < num_experts:
                    experts_in_metadata.add(expert_idx)

        expert_owner: Dict[int, int] = {}
        valid_mapping = True
        for expert_idx in experts_in_metadata:
            owners = [
                src_id
                for src_id in group_ids
                if expert_idx in local_experts_by_device[src_id]
            ]
            if len(owners) != 1:
                valid_mapping = False
                break
            expert_owner[expert_idx] = owners[0]

        if not valid_mapping:
            continue

        for dest_pos, dest_id in enumerate(group_ids):
            dest_output = output_shards[dest_id]
            dest_metadata = group_metadata[dest_id]

            for b_local in range(local_batch):
                global_b = dest_pos * local_batch + b_local
                if global_b >= batch_global:
                    continue
                for s in range(local_seq):
                    for k in range(k_slots):
                        expert_idx = int(dest_metadata[0, global_b, s, k].item())
                        src_id = expert_owner.get(expert_idx, None)
                        if src_id is None:
                            continue

                        src_input = group_inputs[src_id]
                        src_local_experts = local_experts_by_device[src_id]
                        if expert_idx not in src_local_experts:
                            continue

                        local_e = src_local_experts.index(expert_idx)
                        if local_e >= int(src_input.shape[0]):
                            continue
                        src_b = min(b_local, int(src_input.shape[1]) - 1)
                        src_s = min(s, int(src_input.shape[2]) - 1)
                        dest_output[k, b_local, s, :] = src_input[
                            local_e, src_b, src_s, :
                        ]

    return GoldenMapTensor(output_shards, input_tensor.mesh_shape).to(output_dtype)


def ttir_moe_expert_token_remap_golden(
    topk_tensor: GoldenMapTensor,
    expert_mapping: GoldenMapTensor,
    expert_metadata: GoldenMapTensor,
    reduction_size=32,
) -> GoldenMapTensor:
    """Golden for remap. Returns (mapping, reduced) with matching shapes."""
    # topk_tensor: [1, BD, S, E], expert_mapping: [1, 1, E, num_devices]
    E = topk_tensor.shape[3]
    num_devices = expert_mapping.shape[3]
    E_local = E // num_devices
    BD = topk_tensor.shape[1]
    S = topk_tensor.shape[2]
    mesh_shape = topk_tensor.mesh_shape
    num_shards = mesh_shape[0] * mesh_shape[1]
    mapping_tensor = torch.zeros(1, BD, S, E_local, dtype=topk_tensor.dtype)
    mapping = GoldenMapTensor(
        {i: mapping_tensor.clone() for i in range(num_shards)},
        mesh_shape=mesh_shape,
    )
    M = reduction_size
    reduced_seq = (BD * S + M - 1) // M
    reduced_tensor = torch.zeros(1, 1, reduced_seq, E_local, dtype=topk_tensor.dtype)
    reduced = GoldenMapTensor(
        {i: reduced_tensor.clone() for i in range(num_shards)},
        mesh_shape=mesh_shape,
    )
    return mapping, reduced


def _moe_compute_zero_outputs(
    mesh_shape, output_types_mlir, fallback
) -> Tuple[GoldenMapTensor, ...]:
    num_shards = mesh_shape[0] * mesh_shape[1]
    out: List[GoldenMapTensor] = []
    if output_types_mlir is None:
        shapes = [tuple(int(d) for d in fallback.shape)] * 6
        dtypes = [fallback.dtype] * 6
    else:
        shapes = [tuple(int(d) for d in t.shape) for t in output_types_mlir]
        dtypes = [mlir_type_to_torch_dtype(t.element_type) for t in output_types_mlir]
    for shape, dtype in zip(shapes, dtypes):
        placeholder = torch.zeros(shape, dtype=dtype)
        out.append(
            GoldenMapTensor(
                {i: placeholder.clone() for i in range(num_shards)},
                mesh_shape=mesh_shape,
            )
        )
    return tuple(out)


def _swiglu_reference(gate, up, alpha=1.702, clamp_limit=7.0):
    """GPT-OSS SwiGLU activation (tt-metal test_moe_compute_6U._swiglu_reference)."""
    gate_c = torch.clamp(gate, max=clamp_limit)
    up_c = torch.clamp(up, min=-clamp_limit, max=clamp_limit)
    return (up_c + 1.0) * gate_c * torch.sigmoid(alpha * gate_c)


# Blackhole combine-core layout constants (the single-card matmul_output writer).
# max_combine_core_range_set is CoreRange({9,0},{10,7}) and the worker grid is
# 11 wide (moe_compute_program_factory.cpp get_layout / get_moe_combine_cores).
_MOE_BH_GRID_W = 11
_MOE_BH_COMBINE_X = (9, 10)
_MOE_BH_COMBINE_NY = 8


def moe_combine_core_rows(
    height_shard_dim,
    width_shard_dim,
    grid_w=_MOE_BH_GRID_W,
    combine_x=_MOE_BH_COMBINE_X,
    combine_ny=_MOE_BH_COMBINE_NY,
):
    """Host-readback row for each combine core, indexed by j = a*width + b.

    Reproduces tt-metal get_moe_combine_cores (Blackhole) deterministically:
    take corerange_to_cores({9,0}-{10,7}) in its default row_wise=False order
    (x outer, y inner), select the first height*width cores, sort them x-then-y,
    and map each core (x,y) to its host-readback row y*grid_w + x (outputs are
    HEIGHT_SHARDED over the full grid_w x 10 worker grid). For width=4 all 16
    cores are picked either way, but width=3 selects x=9 (y 0..7) + x=10 (y 0..3)
    rather than the first six rows of both columns."""
    cores = [(x, y) for x in combine_x for y in range(combine_ny)]
    cores = sorted(
        cores[: height_shard_dim * width_shard_dim], key=lambda c: (c[0], c[1])
    )
    return [y * grid_w + x for (x, y) in cores]


def moe_combine_scatter_positions(
    active, height_shard_dim, width_shard_dim, wcols, combine_rows
):
    """Yield (t_act, r, dev_t, col0, lo) blocks for one expert's ``active``
    tokens, walking tt-metal's combine-core HEIGHT_SHARDED scatter (dm1.cpp).

    Each block is a ``wcols``-wide slice of the [r, ., dev_t, .] buffer row: the
    active tokens walk height shards (the first ``active % height_shard_dim``
    shards get one extra token), giving (shard a, in-shard row d); width shard b
    selects the combine core j = a*width + b (row combine_rows[j]) and the
    column group, while the in-shard row d packs as device tile-row d//width and
    column (d%width)*wcols. The golden writes mlp[t_act, lo:lo+wcols] into each
    block; the mask writes 1s there."""
    tps, rem = divmod(active, height_shard_dim)
    shard, srow = 0, 0
    for t_act in range(active):
        a, d = shard, srow
        dev_t = d // width_shard_dim
        col0 = (d % width_shard_dim) * wcols
        for b in range(width_shard_dim):
            j = a * width_shard_dim + b
            yield t_act, combine_rows[j], dev_t, col0, b * wcols
        cap = tps + (1 if shard < rem else 0)
        srow += 1
        if srow == cap:
            shard, srow = shard + 1, 0


def ttir_moe_compute_golden(
    tilize_input_tensor: GoldenMapTensor,
    tilize_expert_indices_tensor: GoldenMapTensor,
    tilize_expert_scores_tensor: GoldenMapTensor,
    tilize_expert_mapping_tensor: GoldenMapTensor,
    w0: GoldenMapTensor,
    w1: GoldenMapTensor,
    w2: GoldenMapTensor,
    layer_id=0,
    output_height_shard_dim=0,
    intermediate_size=0,
    has_bias=False,
    cluster_axis=0,
    bias_0: Optional[GoldenMapTensor] = None,
    bias_1: Optional[GoldenMapTensor] = None,
    bias_2: Optional[GoldenMapTensor] = None,
    activation_function=None,
    num_links=None,
    topology=None,
    compute_only=False,
    bh_ring_size=None,
    num_worker_cores=0,
    output_types_mlir: Optional[List[Type]] = None,
) -> Tuple[
    GoldenMapTensor,
    GoldenMapTensor,
    GoldenMapTensor,
    GoldenMapTensor,
    GoldenMapTensor,
    GoldenMapTensor,
]:
    """Golden for moe_compute. Modeled on ``moe_gpt_golden`` (a specialized
    moe_compute variant): reproduces tt-metal's exact byte-packed device output
    layouts so the standard PCC framework can compare directly, after the test
    masks off uninitialized slots with an on-device ``multiply``.

    Outputs 0-2 (per_expert_total_tokens, expert_activation, expert_to_token)
    are routing metadata derived from indices/scores/mapping; their byte
    layouts match ``compute_output_specs`` in moe_compute_device_operation.cpp.
    Outputs 3-4 (tilize_output, matmul_output) are the expert MLP result
    (``act(x@w0) * (x@w1) @ w2`` with act = SiLU or SwiGLU, see
    compute_matmul_golden in test_moe_compute_6U.py) packed into the
    HEIGHT_SHARDED combine-staging
    buffer; in compute_only matmul_output is the final output and
    combine_output (result 5) aliases it.

    Only the compute_only path is modeled; for the full fused path (combine)
    the golden falls back to zeros (verified via tt-metal directly, not here).
    w0/w1/w2 are the raw per-expert weights [L, E_per_device, K, N] (the device
    prepacks them in TTNN; the golden uses the raw values directly). They are
    replicated across devices, so any shard carries the full weight.
    """
    mesh_shape = tilize_input_tensor.mesh_shape

    if not compute_only:
        return _moe_compute_zero_outputs(
            mesh_shape, output_types_mlir, tilize_input_tensor
        )

    # Weights are replicated across the mesh; take any shard.
    raw_w0 = list(w0._shard_map.values())[0]
    raw_w1 = list(w1._shard_map.values())[0]
    raw_w2 = list(w2._shard_map.values())[0]

    hidden_size = int(raw_w0.shape[2])
    if not isinstance(cluster_axis, int):
        cluster_axis = unpack_mlir_attr(cluster_axis)
    act_name = (
        activation_function
        if isinstance(activation_function, (str, type(None)))
        else unpack_mlir_attr(activation_function)
    ) or "silu"

    L1_ALIGN = 16  # l1_alignment on Wormhole/Blackhole

    mapping_tensor = list(tilize_expert_mapping_tensor._shard_map.values())[0]
    mapping_row = mapping_tensor.reshape(-1, mapping_tensor.shape[-1])[0]
    E_total = mapping_row.shape[0]

    grouped_input = tilize_input_tensor.group_by_axis(cluster_axis)
    grouped_idx = tilize_expert_indices_tensor.group_by_axis(cluster_axis)
    grouped_scr = tilize_expert_scores_tensor.group_by_axis(cluster_axis)

    out_tc, out_act, out_et, out_tile, out_tile_rm = {}, {}, {}, {}, {}

    E_per_device = int(raw_w0.shape[1])
    K = hidden_size  # noqa: F841 (kept for parity with moe_gpt naming)

    # MLP HEIGHT_SHARDED buffer geometry (matmul_output result shape).
    tile_shape = (
        tuple(int(d) for d in output_types_mlir[4].shape)
        if output_types_mlir is not None
        else (num_worker_cores, 2, 32, hidden_size)
    )
    TILE = 32
    height_shard_dim = 4
    # tt-metal auto_output_width_shard_dim: largest divisor of (hidden/TILE) <= 4
    # (moe_compute_utils.py).
    width_shard_dim = next(
        (d for d in range(4, 0, -1) if (hidden_size // TILE) % d == 0), 1
    )

    # matmul_output combine-core geometry (Blackhole single card): the host
    # readback row for each combine core j = a*width_shard_dim + b, derived from
    # tt-metal get_moe_combine_cores for this (height, width) shard split.
    combine_rows = moe_combine_core_rows(height_shard_dim, width_shard_dim)
    wcols = hidden_size // width_shard_dim  # cols owned by one width shard

    # Raw per-expert biases [L, E_per_device, .] (None when has_bias is False):
    # b0/b1 broadcast over the intermediate dim, b2 over hidden. The device
    # prepacks them in TTNN; the golden uses the raw values directly.
    raw_b0 = list(bias_0._shard_map.values())[0] if bias_0 is not None else None
    raw_b1 = list(bias_1._shard_map.values())[0] if bias_1 is not None else None
    raw_b2 = list(bias_2._shard_map.values())[0] if bias_2 is not None else None

    def _silu_mlp(x, w0, w1, w2, b0=None, b1=None, b2=None):
        gate = x @ w0
        up = x @ w1
        if b0 is not None:
            gate = gate + b0
        if b1 is not None:
            up = up + b1
        if act_name == "swiglu":
            inter = _swiglu_reference(gate, up)
        else:
            inter = torch.nn.functional.silu(gate) * up
        out = inter @ w2
        if b2 is not None:
            out = out + b2
        return out.to(torch.bfloat16)

    for grp_inp, grp_idx, grp_scr in zip(grouped_input, grouped_idx, grouped_scr):
        dev_ids = sorted(grp_inp.keys())
        ring_devices = len(dev_ids)
        sample = grp_inp[dev_ids[0]]
        total_tokens = sample.reshape(-1, hidden_size).shape[0]
        K_sel = grp_idx[dev_ids[0]].reshape(-1, grp_idx[dev_ids[0]].shape[-1]).shape[-1]
        full_idx = grp_idx[dev_ids[0]].reshape(total_tokens, K_sel)
        full_scr = grp_scr[dev_ids[0]].reshape(total_tokens, K_sel)

        for dev_id in dev_ids:
            local_globals = sorted(
                [e for e in range(E_total) if int(mapping_row[e].item()) == dev_id]
            )[:E_per_device]

            # --- Output 0: per_expert_total_tokens ---
            # tt-metal allocates this HEIGHT_SHARDED across the full worker grid
            # (num_cores rows, one per shard) and multicasts the per-expert counts
            # to every core in the worker bbox, so the same row is replicated on
            # all cores (moe_compute_device_operation.cpp compute_output_specs +
            # the per_expert_total_tokens mcast). Columns past E_per_device are
            # L1-alignment padding (don't-care).
            tc_elements = (E_per_device * 4 + L1_ALIGN - 1) // L1_ALIGN * L1_ALIGN // 4
            num_cores = (
                int(output_types_mlir[0].shape[0])
                if output_types_mlir is not None
                else (num_worker_cores or 1)
            )
            token_counts = torch.zeros(num_cores, tc_elements, dtype=torch.int32)

            # --- Output 1: expert_activation records ---
            # Single INTERLEAVED page sized total_tokens * aligned_row_bytes (NO
            # +1 sentinel row — the sentinel is written in-place at the first
            # unused record slot when fewer than total_tokens tokens activate).
            act_row_stride = (
                ((2 * E_per_device + 1) * 4 + L1_ALIGN - 1) // L1_ALIGN * L1_ALIGN
            ) // 4
            act_total = total_tokens * act_row_stride
            activation_records = torch.zeros(1, act_total, dtype=torch.int32)

            # --- Output 2: expert_to_token ---
            et_entry = (4 + L1_ALIGN - 1) // L1_ALIGN * L1_ALIGN // 4  # = 4
            et_row_elements = (total_tokens + 1) * et_entry
            token_indices = torch.zeros(
                E_per_device, et_row_elements, dtype=torch.int32
            )

            counts = [0] * E_per_device
            act_row_idx = 0
            for t in range(total_tokens):
                activated = False
                row = torch.zeros(2 * E_per_device + 1, dtype=torch.int32)
                row[0] = t
                for le in range(E_per_device):
                    row[1 + le] = K_sel + 1  # sentinel: not selected
                for le, ge in enumerate(local_globals):
                    for k in range(K_sel):
                        if int(full_idx[t, k].item()) == ge:
                            row[1 + le] = k
                            sbits = int.from_bytes(
                                full_scr[t, k]
                                .to(torch.bfloat16)
                                .view(torch.int16)
                                .numpy()
                                .tobytes()[:2],
                                "little",
                            )
                            row[1 + E_per_device + le] = sbits
                            token_indices[le, counts[le] * et_entry] = t
                            counts[le] += 1
                            activated = True
                            break
                if activated:
                    off = act_row_idx * act_row_stride
                    activation_records[0, off : off + (2 * E_per_device + 1)] = row
                    act_row_idx += 1

            s_off = act_row_idx * act_row_stride
            if s_off < act_total:
                activation_records[0, s_off] = -1
            for e in range(E_per_device):
                token_counts[:, e] = counts[e]
                sp = counts[e] * et_entry
                if sp < et_row_elements:
                    token_indices[e, sp] = -1

            out_tc[dev_id] = token_counts
            out_act[dev_id] = activation_records
            out_et[dev_id] = token_indices

            # --- Outputs 3-4: SiLU MLP scattered into the matmul writer's
            # HEIGHT_SHARDED combine-core layout (matmul_output). This is the
            # exact forward of tt-metal's combine-core scatter (dm1.cpp) +
            # prepare_output_tensor_from_combine_writer / validate_matmul
            # (tests/nightly/tg/ccl/moe/test_moe_compute_6U.py), inverted:
            #   * buffer slot c == local expert le (E_per_device==2 double buffer,
            #     [.,2,32,.] axis; experts_to_check = [(0,0),(1,1)]).
            #   * per expert the active tokens walk height shards: the first
            #     active%height_shard_dim shards get one extra token, giving
            #     (shard a, in-shard row d) for token t_act.
            #   * width shard b = hid//wcols selects the combine-core column
            #     group; combine core index j = a*width_shard_dim + b maps to the
            #     host-readback row r = (j%COMBINE_NY)*GRID_W + COMBINE_X0 + j//COMBINE_NY.
            #   * within the shard the in-shard row d packs as device tile-row
            #     d//width_shard_dim and column (d%width_shard_dim)*wcols + f.
            sparse_in = grp_inp[dev_id].reshape(total_tokens, hidden_size)
            tile_golden = torch.zeros(tile_shape, dtype=torch.bfloat16)
            num_buffers = tile_shape[1]  # 2 (double buffer)
            for le, ge in enumerate(local_globals):
                if le >= num_buffers:
                    break
                toks = [
                    sparse_in[t]
                    for t in range(total_tokens)
                    if any(int(full_idx[t, k].item()) == ge for k in range(K_sel))
                ]
                if not toks:
                    continue
                x = torch.stack(toks, dim=0).float()
                mlp = _silu_mlp(
                    x,
                    raw_w0[0, le].float(),
                    raw_w1[0, le].float(),
                    raw_w2[0, le].float(),
                    raw_b0[0, le].float() if raw_b0 is not None else None,
                    raw_b1[0, le].float() if raw_b1 is not None else None,
                    raw_b2[0, le].float() if raw_b2 is not None else None,
                )
                active = mlp.shape[0]
                for t_act, r, dev_t, col0, lo in moe_combine_scatter_positions(
                    active, height_shard_dim, width_shard_dim, wcols, combine_rows
                ):
                    tile_golden[r, le, dev_t, col0 : col0 + wcols] = mlp[
                        t_act, lo : lo + wcols
                    ]

            out_tile[dev_id] = tile_golden
            out_tile_rm[dev_id] = tile_golden.clone()

    return (
        GoldenMapTensor(out_tc, mesh_shape),
        GoldenMapTensor(out_act, mesh_shape),
        GoldenMapTensor(out_et, mesh_shape),
        GoldenMapTensor(out_tile, mesh_shape),
        GoldenMapTensor(out_tile_rm, mesh_shape),
        GoldenMapTensor(out_tile_rm, mesh_shape),  # combine_output aliases matmul
    )


def ttir_matmul_golden(
    a: GoldenMapTensor,
    b: GoldenMapTensor,
    transpose_a_attr: BoolAttr,
    transpose_b_attr: BoolAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    transpose_a = unpack_mlir_attr(transpose_a_attr)
    transpose_b = unpack_mlir_attr(transpose_b_attr)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    a = torch.transpose(a, -2, -1) if transpose_a else a
    b = torch.transpose(b, -2, -1) if transpose_b else b
    return torch.matmul(a, b).to(output_dtype)


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


def sdpa_decode_golden(
    query: GoldenMapTensor,
    key: GoldenMapTensor,
    value: GoldenMapTensor,
    cur_pos_tensor: GoldenMapTensor = None,
    attention_mask=None,
    is_causal=True,
    scale=None,
    **kwargs,
) -> GoldenMapTensor:
    """
    Golden function for scaled dot product attention decode.
    Matches tt-metal's Flash-Decode implementation.

    Decode layout:
        Query:  [1, batch, num_heads, head_dim]
        Key:    [batch, num_kv_heads, seq_len, head_dim]
        Value:  [batch, num_kv_heads, seq_len, head_dim]
        Mask:   [batch_or_1, 1, num_heads_or_1, seq_len]
        Output: [1, batch, num_heads, head_dim]
    """
    # Query: [1, B, H, D] -> [B, H, 1, D]
    q = query.float().squeeze(0).unsqueeze(2)

    k = key.float()
    v = value.float()

    q_heads = q.shape[1]
    kv_heads = k.shape[1]

    # Handle GQA: broadcast K/V heads to match Q heads
    if q_heads != kv_heads:
        assert q_heads % kv_heads == 0
        num_repeats = q_heads // kv_heads
        k = torch.repeat_interleave(k, num_repeats, dim=1)
        v = torch.repeat_interleave(v, num_repeats, dim=1)

    # QK = Q @ K^T: [B, H, 1, D] @ [B, H, D, S] -> [B, H, 1, S]
    qk = torch.matmul(q, k.transpose(-2, -1))

    # Add attention mask (before scaling, matching tt-metal)
    # Mask is in decode layout [B, 1, H, S], permute to [B, H, 1, S] to match qk
    if attention_mask is not None:
        qk = torch.add(qk, attention_mask.float().permute(0, 2, 1, 3))
    elif is_causal and cur_pos_tensor is not None:
        # Synthesize a per-batch causal mask: positions j > cur_pos[b] are -inf.
        b, _, _, seq_len = qk.shape
        causal_mask = torch.zeros((b, 1, 1, seq_len), dtype=torch.float32)
        cur_t = _gmt_leaf_torch(cur_pos_tensor)
        for i in range(b):
            start_idx = int(cur_t[i].item())
            causal_mask[i, :, :, start_idx + 1 :] = float("-inf")
        qk = torch.add(qk, causal_mask)

    # Scale AFTER masking (tt-metal fuses scale into exp)
    if scale is not None:
        qk = torch.mul(qk, scale)

    # Softmax + matmul with V
    attn_weights = torch.softmax(qk, dim=-1)
    output = torch.matmul(attn_weights, v)  # [B, H, 1, D]

    # Output: [B, H, 1, D] -> [1, B, H, D]
    output = output.squeeze(2).unsqueeze(0)

    return output.to(query.dtype)


def stablehlo_dot_general_golden(
    lhs: GoldenMapTensor,
    rhs: GoldenMapTensor,
    batch_dims_lhs,
    contract_dims_lhs,
    batch_dims_rhs,
    contract_dims_rhs,
) -> GoldenMapTensor:
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


def ttir_embedding_golden(
    indices_tensor: GoldenMapTensor,
    weight_tensor: GoldenMapTensor,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    vocab_size = weight_tensor.size(-2)
    embed_dim = weight_tensor.size(-1)
    weight_2d = weight_tensor.reshape(vocab_size, embed_dim)
    embedding = torch.nn.Embedding.from_pretrained(weight_2d)
    golden_typecast = indices_tensor.to(torch.int32)
    golden_input = torch.clamp(golden_typecast, 0, (vocab_size - 1))
    return embedding(golden_input).to(output_dtype)


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

    begin = begin if begin >= 0 else input_tensor.size(dim) + begin
    end = end if end >= 0 else input_tensor.size(dim) + end
    num_indices = math.ceil((end - begin) / step)
    indices = []
    for i in range(num_indices):
        indices.append((begin + i) * step)
    index = torch.tensor(indices)
    return torch.index_select(input_tensor, dim=dim, index=index)


def gather_golden(
    input_tensor: GoldenMapTensor,
    start_indices_tensor: GoldenMapTensor,
    dimension_numbers,
    slice_sizes: DenseI64ArrayAttr,
    indices_are_sorted: BoolAttr,
) -> GoldenMapTensor:
    """Golden function for stablehlo.gather operation.

    Performs a gather on ``input_tensor`` according to ``start_indices_tensor``
    and the StableHLO gather ``dimension_numbers``.
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
    offset_dims = list(dimension_numbers.offset_dims)
    collapsed_slice_dims = list(dimension_numbers.collapsed_slice_dims)
    operand_batching_dims = list(dimension_numbers.operand_batching_dims)
    start_indices_batching_dims = list(dimension_numbers.start_indices_batching_dims)
    start_index_map = list(dimension_numbers.start_index_map)
    index_vector_dim = dimension_numbers.index_vector_dim
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
    # Two patterns this golden supports:
    # - Embedding-style: collapsed_slice_dims == start_index_map (all indexed
    #   dims have slice size 1 and are removed from the output).
    # - Window-style: collapsed_slice_dims is empty (the indexed dims keep their
    #   slice as a window in the output).
    assert (
        set(collapsed_slice_dims) == set(start_index_map)
        or len(collapsed_slice_dims) == 0
    ), (
        "gather golden assumes collapsed_slice_dims == start_index_map (embedding-"
        "style) or collapsed_slice_dims is empty (window-style)"
    )
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

    # desired_index_for_current[i] = j says current axis i should end up at
    # output axis j (source -> target). torch.permute expects the inverse
    # (output[i] = input[perm[i]], i.e. target -> source), so invert before
    # passing it in.
    inverse_perm = [0] * result_rank
    for src, tgt in enumerate(desired_index_for_current):
        inverse_perm[tgt] = src

    if inverse_perm != list(range(result_rank)):
        gathered = gathered.permute(*inverse_perm)

    return gathered.to(device=device)


def stablehlo_scatter_golden(
    inputs: List[GoldenMapTensor],
    scatter_indices: GoldenMapTensor,
    updates: List[GoldenMapTensor],
    scatter_dimension_numbers,
    update_computation_region,
    result_types: List[Type],
) -> Union[GoldenMapTensor, List[GoldenMapTensor]]:
    # Unpack dimension numbers
    update_window_dims = list(scatter_dimension_numbers.update_window_dims)
    inserted_window_dims = list(scatter_dimension_numbers.inserted_window_dims)
    input_batching_dims = list(scatter_dimension_numbers.input_batching_dims)
    scatter_indices_batching_dims = list(
        scatter_dimension_numbers.scatter_indices_batching_dims
    )
    scattered_dims_to_operand_dims = list(
        scatter_dimension_numbers.scattered_dims_to_operand_dims
    )
    index_vector_dim = scatter_dimension_numbers.index_vector_dim

    # Currently support simple cases without batching
    assert len(input_batching_dims) == 0, "input_batching_dims not supported yet"
    assert (
        len(scatter_indices_batching_dims) == 0
    ), "scatter_indices_batching_dims not supported yet"

    # Validate update_window_dims are trailing dimensions (as currently implemented)
    # This golden assumes update_window_dims are the trailing dimensions of the update tensor
    # in order, which is the common case for most scatter operations.
    if len(update_window_dims) > 0:
        expected_trailing = list(
            range(
                len(updates[0].shape) - len(update_window_dims), len(updates[0].shape)
            )
        )
        assert update_window_dims == expected_trailing, (
            f"scatter golden currently only supports update_window_dims as trailing "
            f"dimensions. Got update_window_dims={update_window_dims}, but expected "
            f"{expected_trailing} for update shape {updates[0].shape}. "
            f"Arbitrary update_window_dims require proper dimension mapping."
        )

    # Work with first shard for replicated tensors
    def _first_shard(t):
        return t.shard_at(0) if isinstance(t, GoldenMapTensor) else t

    # Initialize outputs as clones of inputs
    outputs = [inp.clone() for inp in inputs]

    # Get tensor shapes
    scatter_indices_shape = list(_first_shard(scatter_indices).shape)

    # Determine the batch shape (indices without index_vector_dim)
    if index_vector_dim < len(scatter_indices_shape):
        batch_shape = (
            scatter_indices_shape[:index_vector_dim]
            + scatter_indices_shape[index_vector_dim + 1 :]
        )
        index_depth = scatter_indices_shape[index_vector_dim]
    else:
        # index_vector_dim == rank means each element is a scalar index
        batch_shape = scatter_indices_shape
        index_depth = 1

    # Flatten scatter indices to [num_indices, index_depth]
    num_indices = int(torch.tensor(batch_shape).prod()) if batch_shape else 1
    indices_flat = (
        _first_shard(scatter_indices).reshape(num_indices, index_depth).long()
    )

    # Bounds check indices before processing (similar to gather_golden)
    for i in range(len(inputs)):
        input_shard = _first_shard(inputs[i])
        for k, operand_dim in enumerate(scattered_dims_to_operand_dims):
            if operand_dim in inserted_window_dims:
                # For inserted dims, the index must be in valid range
                valid_max = input_shard.size(operand_dim) - 1
            else:
                # For window dims, the index + window size must fit
                # Find the corresponding update window dimension
                update_dim_for_operand = None
                non_inserted_count = 0
                for d in range(len(input_shard.shape)):
                    if d == operand_dim and d not in inserted_window_dims:
                        if non_inserted_count < len(update_window_dims):
                            update_dim_for_operand = update_window_dims[
                                non_inserted_count
                            ]
                        break
                    if d not in inserted_window_dims:
                        non_inserted_count += 1

                if update_dim_for_operand is not None:
                    update_shard = _first_shard(updates[i])
                    window_size = update_shard.shape[update_dim_for_operand]
                    valid_max = input_shard.size(operand_dim) - window_size
                else:
                    valid_max = input_shard.size(operand_dim) - 1

            if k < index_depth:
                if torch.any(indices_flat[:, k] < 0) or torch.any(
                    indices_flat[:, k] > valid_max
                ):
                    raise IndexError(
                        f"scatter indices out of bounds for operand {i} dim {operand_dim}: "
                        f"indices range [{indices_flat[:, k].min()}, {indices_flat[:, k].max()}], "
                        f"valid range [0, {valid_max}]"
                    )

    # Process each scatter index
    for idx_num in range(num_indices):
        # Get the index vector for this scatter location
        index_vector = indices_flat[idx_num]

        # Build the start indices for the operand
        operand_start_indices = [0] * len(scattered_dims_to_operand_dims)
        for i, operand_dim in enumerate(scattered_dims_to_operand_dims):
            if i < len(index_vector):
                operand_start_indices[i] = int(index_vector[i].item())

        # For each input/update pair
        for i in range(len(inputs)):
            output_tensor = outputs[i]
            if len(output_tensor.shard_map) > 1:
                first_output_shard = _first_shard(output_tensor)
                for device, shard in output_tensor.shard_map.items():
                    if not torch.equal(shard, first_output_shard):
                        raise AssertionError(
                            "stablehlo_scatter_golden expects replicated output shards when "
                            "applying scatter via _first_shard(outputs[i])"
                        )
                for device in output_tensor.shard_map:
                    output_tensor.shard_map[device] = first_output_shard
            input_shard = _first_shard(output_tensor)

            update_tensor = updates[i]
            if len(update_tensor.shard_map) > 1:
                first_update_shard = _first_shard(update_tensor)
                for device, shard in update_tensor.shard_map.items():
                    if not torch.equal(shard, first_update_shard):
                        raise AssertionError(
                            "stablehlo_scatter_golden expects replicated update shards when "
                            "reading updates via _first_shard(updates[i])"
                        )
            update_shard = _first_shard(update_tensor)
            update_shape = list(update_shard.shape)

            # Calculate the update slice for this index
            # The update tensor has batch dims followed by window dims
            update_batch_idx = []
            if num_indices > 1:
                # Convert flat index back to multi-dimensional batch index
                remaining = idx_num
                for dim_size in reversed(batch_shape):
                    update_batch_idx.insert(0, remaining % dim_size)
                    remaining //= dim_size

            # Extract the update window for this batch element
            update_slice_indices = tuple(update_batch_idx) + (slice(None),) * len(
                update_window_dims
            )
            update_window = (
                update_shard[update_slice_indices] if update_batch_idx else update_shard
            )

            # Build the scatter slice in the operand
            # Map update window to operand, inserting dims as needed
            operand_indices = []
            update_dim_idx = 0
            for operand_dim in range(len(input_shard.shape)):
                if operand_dim in inserted_window_dims:
                    # This dim is collapsed in update, use index
                    mapped_idx = (
                        scattered_dims_to_operand_dims.index(operand_dim)
                        if operand_dim in scattered_dims_to_operand_dims
                        else None
                    )
                    if mapped_idx is not None and mapped_idx < len(
                        operand_start_indices
                    ):
                        operand_indices.append(operand_start_indices[mapped_idx])
                    else:
                        operand_indices.append(0)
                else:
                    # This dim exists in update window
                    operand_indices.append(slice(None))
                    update_dim_idx += 1

            # Apply the update computation
            # Bounds should already be validated, so use operand_indices directly
            if len(operand_indices) > 0:
                try:
                    current_value = input_shard[tuple(operand_indices)]
                except IndexError as e:
                    raise IndexError(
                        f"Failed to index operand {i} with indices {operand_indices} "
                        f"(operand shape: {input_shard.shape}, scatter index {idx_num}): {e}"
                    )

                # Check if the update computation is a simple replacement or an actual computation
                # by looking at the region operations
                is_simple_replacement = True
                applied_op = False
                if update_computation_region is not None:
                    for block in update_computation_region.blocks:
                        for op in block.operations:
                            # Check if there's any computation beyond just returning the update value
                            op_type = type(op).__name__
                            if hasattr(op, "OPERATION_NAME"):
                                op_name = op.OPERATION_NAME
                            else:
                                op_name = op_type

                            # Skip return operations
                            if "return" in str(op_name).lower():
                                continue

                            is_simple_replacement = False
                            # Try to determine the operation type
                            try:
                                if "AddOp" in op_type or "add" in str(op_name).lower():
                                    input_shard[tuple(operand_indices)] = (
                                        current_value + update_window
                                    )
                                    applied_op = True
                                elif (
                                    "MulOp" in op_type
                                    or "mul" in str(op_name).lower()
                                    or "multiply" in str(op_name).lower()
                                ):
                                    input_shard[tuple(operand_indices)] = (
                                        current_value * update_window
                                    )
                                    applied_op = True
                                elif (
                                    "MaxOp" in op_type or "max" in str(op_name).lower()
                                ):
                                    input_shard[tuple(operand_indices)] = torch.maximum(
                                        current_value, update_window
                                    )
                                    applied_op = True
                                elif (
                                    "MinOp" in op_type or "min" in str(op_name).lower()
                                ):
                                    input_shard[tuple(operand_indices)] = torch.minimum(
                                        current_value, update_window
                                    )
                                    applied_op = True
                                else:
                                    # Default to replacement if we don't recognize the op
                                    input_shard[tuple(operand_indices)] = update_window
                                    applied_op = True
                            except (IndexError, RuntimeError) as e:
                                raise RuntimeError(
                                    f"Failed to apply update computation for operand {i}, "
                                    f"scatter index {idx_num}: update shape {update_window.shape}, "
                                    f"current value shape {current_value.shape}, error: {e}"
                                )

                            if applied_op:
                                break
                        if applied_op:
                            break

                if is_simple_replacement and not applied_op:
                    # Simple replacement (assumes stablehlo.return %arg1 in region)
                    try:
                        input_shard[tuple(operand_indices)] = update_window
                    except (IndexError, RuntimeError) as e:
                        raise RuntimeError(
                            f"Failed to apply replacement for operand {i}, "
                            f"scatter index {idx_num}: update shape {update_window.shape}, "
                            f"operand indices {operand_indices}, error: {e}"
                        )

    # Return outputs with proper types
    output_results = []
    for i, output in enumerate(outputs):
        if i < len(result_types):
            output_dtype = mlir_type_to_torch_dtype(
                result_types[i].element_type
                if hasattr(result_types[i], "element_type")
                else result_types[i]
            )
            output_results.append(output.to(output_dtype))
        else:
            output_results.append(output)

    return output_results[0] if len(output_results) == 1 else output_results


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
    batch_offset=None,
    output_type_mlir=None,
    **kwargs,
) -> GoldenMapTensor:
    """
    Custom golden function for update_cache operation.

    Matches TTNN ``update_cache`` semantics:
      cache shape: [num_users, num_heads, max_seq_len, head_dim]
      input shape: [1, num_heads, num_input_users, head_dim]
      update_index: scalar position in the cache sequence dimension
    The update writes:
      cache[batch_offset + b, h, update_index, d] = input[0, h, b, d]
    for ``b`` in ``[0, num_input_users)``.

    Parameters
    ----------
    cache_tensor : GoldenMapTensor
        Cache tensor to update
    update_tensor : GoldenMapTensor
        Tensor containing update data
    indices_tensor : GoldenMapTensor
        Tensor containing the (scalar) update index
    batch_offset : IntegerAttr, optional
        Offset along the cache batch dimension where the input batch starts
    output_type_mlir : Type, optional
        MLIR output type (ignored in golden computation)
    **kwargs : dict
        Additional keyword arguments

    Returns
    -------
    GoldenMapTensor
        Updated cache tensor
    """
    result = cache_tensor.clone()

    indices = indices_tensor.shard_at(0).to(torch.long)
    update_idx = int(indices.flatten()[0].item())

    batch_offset_val = (
        int(unpack_mlir_attr(batch_offset)) if batch_offset is not None else 0
    )

    for device_id, shard in result.shard_map.items():
        update_data = update_tensor.shard_at(device_id)
        num_input_users = update_data.shape[2]
        # update_data[0, h, b, d] -> shard[batch_offset + b, h, update_idx, d]
        shard[
            batch_offset_val : batch_offset_val + num_input_users,
            :,
            update_idx,
            :,
        ] = update_data[0].permute(1, 0, 2)
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
    # torch.mean requires floating point input, cast if needed.
    if not input_tensor.is_floating_point():
        input_tensor = input_tensor.to(torch.float32)
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
    Golden function for softmax with TTIR/TTNN parameter names.

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
    dimension = kwargs.get("dimension", 1)
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

    size = input_tensor.size(dim)
    begin = begin if begin >= 0 else size + begin
    end = end if end >= 0 else size + end
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


def stablehlo_not_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)

    if input_tensor.dtype == torch.bool:
        result = torch.logical_not(input_tensor)
    else:
        result = torch.bitwise_not(input_tensor)

    return result.to(output_dtype)


################ Golden Utilities ###############


def apply_sharding(
    tensor: GoldenMapTensor,
    mesh_shape: Tuple[int],
    shard_dims: Tuple[Union[int, None]],
) -> GoldenMapTensor:
    shards = [tensor.shard_at(0).clone()]
    if len(mesh_shape) != len(shard_dims):
        raise ValueError("mesh_shape and shard_dims must have the same length")
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


def ttir_cumprod_golden(
    input_tensor: GoldenMapTensor, dim: IntegerAttr, output_type_mlir: Type
) -> GoldenMapTensor:
    dim = unpack_mlir_attr(dim)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.cumprod(input_tensor, dim=dim).to(output_dtype)


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


def ttir_gelu_backward_golden(grad, input, approximate="none"):
    # torch.ops.aten.gelu_backward with approximate="none" does not support
    # implicit broadcasting (ONEDNN limitation). Broadcast inputs explicitly.
    grad, input = torch.broadcast_tensors(grad, input)
    return torch.ops.aten.gelu_backward(grad, input, approximate=approximate)


def ttir_gelu_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.nn.functional.gelu(input_tensor).to(output_dtype)


def ttir_cos_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.cos(input_tensor).to(output_dtype)


def ttir_acos_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.acos(input_tensor).to(output_dtype)


def ttir_asin_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.asin(input_tensor).to(output_dtype)


def ttir_asinh_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.asinh(input_tensor).to(output_dtype)


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


def ttir_square_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.square(input_tensor).to(output_dtype)


def ttir_exp2_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.exp2(input_tensor).to(output_dtype)


def ttir_softsign_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.nn.functional.softsign(input_tensor).to(output_dtype)


def ttir_signbit_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    """Match TTKernel signbit_tile: 0.0 or 1.0 in the output element type."""
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.signbit(input_tensor).to(output_dtype)


def ttir_frac_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.frac(input_tensor).to(output_dtype)


def ttir_trunc_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.trunc(input_tensor).to(output_dtype)


def ttir_selu_golden(
    input_tensor: GoldenMapTensor,
    scale_attr: FloatAttr,
    alpha_attr: FloatAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    scale = unpack_mlir_attr(scale_attr)
    alpha = unpack_mlir_attr(alpha_attr)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    pos = torch.clamp(input_tensor, min=0)
    exp_m1 = torch.sub(torch.exp(input_tensor), 1.0)
    neg = torch.clamp(torch.mul(exp_m1, alpha), max=0)
    return torch.mul(torch.add(pos, neg), scale).to(output_dtype)


def ttir_pow_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.pow(input_tensor, other_tensor).to(output_dtype)


def ttir_atan2_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.atan2(input_tensor, other_tensor).to(output_dtype)


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


def ttir_bitwise_or_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.bitwise_or(input_tensor, other_tensor).to(output_dtype)


def ttir_bitwise_xor_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.bitwise_xor(input_tensor, other_tensor).to(output_dtype)


def ttir_bitwise_not_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.bitwise_not(input_tensor).to(output_dtype)


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


def ttir_logical_left_shift_golden(
    input_tensor: GoldenMapTensor, shift_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    input_int64 = input_tensor.to(torch.int64)
    shift_int64 = shift_tensor.to(torch.int64)
    input_unsigned = torch.bitwise_and(input_int64, 0xFFFFFFFF)
    result = torch.bitwise_left_shift(input_unsigned, shift_int64)
    return torch.bitwise_and(result, 0xFFFFFFFF).to(output_dtype)


def ttir_right_shift_golden(
    input_tensor: GoldenMapTensor, shift_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.bitwise_right_shift(input_tensor, shift_tensor).to(output_dtype)


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
        shard_map[device_id] = shard[tuple(slices)]

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
    value_attr: DenseElementsAttr, mesh_shape_attr: ArrayAttr
) -> GoldenMapTensor:
    shape = list(value_attr.type.shape)
    mesh_shape = unpack_mlir_attr(mesh_shape_attr)
    dtype = mlir_type_to_torch_dtype(value_attr.type.element_type)

    if value_attr.is_splat:
        value_attr = value_attr.get_splat_value()
        torch_tensor = torch.full(shape, value_attr.value, dtype=dtype)
    else:
        # PyTorch bfloat16 is packed as uint16 bits in DenseElementsAttr
        # MLIR's Python bindings don't support np.array() on bf16 DenseElementsAttr
        # Extract the hex-encoded data instead
        if dtype == torch.bfloat16:
            attr_str = str(value_attr)
            # MLIR uses hex encoding for large tensors: dense<"0xHEXDATA"> : tensor<...xbf16>
            match = re.search(r'"0x([0-9A-F]+)"', attr_str, re.IGNORECASE)
            if match:
                hex_str = match.group(1)
                byte_data = bytes.fromhex(hex_str)
                u16_array = np.frombuffer(byte_data, dtype=np.uint16)
                torch_tensor = (
                    torch.from_numpy(u16_array.astype(np.int16))
                    .view(torch.bfloat16)
                    .reshape(shape)
                )
            else:
                # Small tensors might use dense<[[value_attr, ...]]> format
                # Parse the float values and convert to bfloat16
                raise NotImplementedError(
                    f"Non-hex bfloat16 constant not yet supported: {attr_str[:100]}"
                )
        else:
            torch_tensor = torch.tensor(np.array(value_attr), dtype=dtype).reshape(
                shape
            )

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


def ttir_argmax_golden(
    input_tensor: GoldenMapTensor,
    dim_arg_attr: ArrayAttr,
    keep_dim_attr: BoolAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    dim_arg = None if dim_arg_attr is None else unpack_mlir_attr(dim_arg_attr)
    keep_dim = unpack_mlir_attr(keep_dim_attr)

    if isinstance(dim_arg, int):
        dim_arg = [dim_arg]

    if dim_arg is None:
        result = torch.argmax(input_tensor, keepdim=keep_dim)
    elif len(dim_arg) == 1:
        result = torch.argmax(input_tensor, dim=dim_arg[0], keepdim=keep_dim)
    else:
        all_dims = list(range(input_tensor.dim()))
        reduce_dims = dim_arg
        non_reduce_dims = [d for d in all_dims if d not in reduce_dims]
        perm_order = non_reduce_dims + reduce_dims
        permuted = input_tensor.permute(*perm_order)

        reduce_size = 1
        for d in reduce_dims:
            reduce_size *= input_tensor.size(d)

        non_reduce_shape = [input_tensor.size(d) for d in non_reduce_dims]
        reshaped = permuted.reshape(*non_reduce_shape, reduce_size)
        result_flat = torch.argmax(reshaped, dim=-1)

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


def ttir_clamp_scalar_golden(
    input_tensor: GoldenMapTensor,
    min_attr,
    max_attr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    min_val = unpack_mlir_attr(min_attr)
    max_val = unpack_mlir_attr(max_attr)
    return torch.clamp(input_tensor, min=min_val, max=max_val).to(output_dtype)


def ttir_logical_or_golden(
    input_tensor: GoldenMapTensor,
    other_tensor: GoldenMapTensor,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.logical_or(input_tensor, other_tensor).to(output_dtype)


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


def ttir_gather_golden(
    input_tensor: GoldenMapTensor,
    index: GoldenMapTensor,
    dim: IntegerAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    dim_value = unpack_mlir_attr(dim)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    index_copy = index.clone()
    index_copy = index_copy.to(torch.int64)
    out_tensor = torch.gather(input_tensor, dim_value, index_copy)
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
    input_tensors,
    dimension_attr: IntegerAttr,
    is_stable_attr: BoolAttr,
    descending_attr: BoolAttr,
    output_types_mlir,
):
    # Normalize single-input case to tuple
    if isinstance(input_tensors, GoldenMapTensor):
        input_tensors = (input_tensors,)
    if not isinstance(output_types_mlir, (list, tuple)):
        output_types_mlir = [output_types_mlir]

    dimension = unpack_mlir_attr(dimension_attr)
    is_stable = unpack_mlir_attr(is_stable_attr)
    descending = unpack_mlir_attr(descending_attr)

    key_tensor = input_tensors[0]
    sorted_key, indices = torch.sort(
        key_tensor, dim=dimension, descending=descending, stable=is_stable
    )

    results = [sorted_key.to(mlir_type_to_torch_dtype(output_types_mlir[0]))]

    for i in range(1, len(input_tensors)):
        sorted_value = torch.gather(input_tensors[i], dimension, indices)
        results.append(sorted_value.to(mlir_type_to_torch_dtype(output_types_mlir[i])))

    return tuple(results)


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


def ttir_gelu_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.nn.functional.gelu(input_tensor).to(output_dtype)


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


def ttir_mesh_partition_golden(
    input: GoldenMapTensor,
    dim_attr: IntegerAttr,
    cluster_axis_attr: IntegerAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    # `mesh_partition` is the lowering of `sdy.all_slice`: along `cluster_axis`
    # the input is replicated, and we split it into N pieces along `dim` (where
    # N is the mesh size on `cluster_axis`), distributing one slice to each
    # device in the group. We assume the per-group shards are identical
    # (replicated) and slice the first one. The input may also arrive fully
    # replicated (single shard) when wrapped with `MeshShardType.Identity`, in
    # which case we treat that single tensor as the per-group source.
    dim = unpack_mlir_attr(dim_attr)
    cluster_axis = unpack_mlir_attr(cluster_axis_attr)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)

    rows, cols = input.mesh_shape
    shard_map = input.shard_map
    fallback_shard = next(iter(shard_map.values()))

    cluster_size = cols if cluster_axis == 1 else rows
    output_shards: Dict[int, torch.Tensor] = {}
    for r in range(rows):
        for c in range(cols):
            device_id = r * cols + c
            group_index = c if cluster_axis == 1 else r
            source = shard_map.get(device_id, fallback_shard)
            scattered = torch.chunk(source, cluster_size, dim=dim)
            output_shards[device_id] = scattered[group_index].clone().to(output_dtype)

    return GoldenMapTensor(output_shards, input.mesh_shape)


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

    # Always produce sorted output for golden comparison. When sorted=False,
    # the order of returned elements is implementation-defined, so both torch
    # and the hardware are free to return any ordering. The golden comparison
    # uses element-wise PCC which requires positional correspondence, so we
    # must normalize the order.
    values, indices = torch.topk(
        input_tensor, k=k, dim=dim, largest=largest, sorted=True
    )

    return values.to(output_dtype), indices.to(torch.uint16)


def ttir_topk_router_gpt_golden(
    input_tensor: GoldenMapTensor,
    weight_tensor: GoldenMapTensor,
    bias_tensor: GoldenMapTensor,
    k_attr: IntegerAttr,
    _num_experts_attr: IntegerAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    k = unpack_mlir_attr(k_attr)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)

    # Fused linear projection: router_logits = input @ weight + bias
    # Use bfloat16 to match device precision and preserve top-k ordering.
    router_logits = torch.matmul(
        input_tensor.to(torch.bfloat16), weight_tensor.to(torch.bfloat16)
    )
    router_logits = router_logits + bias_tensor.to(torch.bfloat16)

    # Select top-k experts.  Output shape is [B, k] (the semantic shape).
    # The k_padded hardware constraint is handled by a TTNN workaround pass.
    topk_values, topk_indices = torch.topk(
        router_logits, k=k, dim=-1, largest=True, sorted=True
    )

    # Softmax over the top-k logits (matches kernel: cols k..31 are masked to
    # -inf before softmax, so only the top-k positions contribute).
    expert_weights = torch.softmax(topk_values, dim=-1)

    expert_indices = topk_indices.to(torch.uint16)
    expert_weights = expert_weights.to(output_dtype)

    return expert_indices, expert_weights


def ttnn_sampling_golden(
    input_values: GoldenMapTensor,
    input_indices: GoldenMapTensor,
    _k: GoldenMapTensor,
    _p: GoldenMapTensor,
    temp: GoldenMapTensor,
    _seed: Optional[IntegerAttr],
    output_type_mlir: Type,
) -> GoldenMapTensor:
    """CPU golden for ttnn.sampling (fused top-k/p + multinomial).

    Inputs are already pre-filtered candidates, so k/p are unused on the
    CPU side. The device kernel is stochastic with hardware RNG that
    cannot be mirrored on CPU, so callers should disable PCC comparison.
    """
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    temperature = temp.float().clamp(min=1e-6).unsqueeze(-1)
    scaled = torch.div(input_values.float(), temperature)
    probs = torch.softmax(scaled, dim=-1)
    sampled_local = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return (
        torch.gather(input_indices, 1, sampled_local.unsqueeze(-1))
        .squeeze(-1)
        .to(output_dtype)
    )


def ttir_sampling_golden(
    input_values: GoldenMapTensor,
    input_indices: GoldenMapTensor,
    k: GoldenMapTensor,
    p: GoldenMapTensor,
    temp: GoldenMapTensor,
    seed: Optional[IntegerAttr],
    output_type_mlir: Type,
) -> GoldenMapTensor:
    """CPU golden for ttir.sampling.

    ttir.sampling lowers 1:1 to ttnn.sampling, so the reference is identical;
    delegate to ttnn_sampling_golden. Stochastic on device (hardware RNG), so
    callers should disable PCC comparison.
    """
    return ttnn_sampling_golden(
        input_values, input_indices, k, p, temp, seed, output_type_mlir
    )


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


def stablehlo_broadcast_in_dim_golden(
    input_tensor: GoldenMapTensor,
    broadcast_dimensions_attr: DenseI64ArrayAttr,
    output_shape: List[int],
    output_type_mlir: Type,
) -> GoldenMapTensor:
    broadcast_dimensions = unpack_mlir_attr(broadcast_dimensions_attr)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)

    # broadcast_dimensions specifies which dimensions of the output correspond to input dimensions
    # We need to reshape the input to match the output rank first, then broadcast
    input_shape = list(input_tensor.shape)

    # Create a shape with 1s for all dimensions, then fill in the input dimensions
    expanded_shape = [1] * len(output_shape)
    for i, dim_idx in enumerate(broadcast_dimensions):
        expanded_shape[dim_idx] = input_shape[i]

    # Reshape input to the expanded shape
    reshaped = input_tensor.reshape(expanded_shape)

    # Now broadcast to the target shape
    result = torch.broadcast_to(reshaped, output_shape)
    return result.to(output_dtype)


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


def stablehlo_sign_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.sign(input_tensor).to(output_dtype)


def stablehlo_convert_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return input_tensor.to(output_dtype)


def stablehlo_composite_golden(
    *operand_tensors: GoldenMapTensor,
    decomposition_fn=None,
    **_kwargs,
) -> GoldenMapTensor:
    """
    Golden for ``stablehlo.composite``.

    A composite op is semantically equivalent to a call into its referenced
    decomposition ``func.func``. This helper walks the decomposition body and
    dispatches every inner op through the same ``GOLDEN_MAPPINGS`` table so
    each decomposition is golden'd exactly the way it would be if it were
    inlined at the composite call site. The ``decomposition_fn`` keyword must
    be the ``func.FuncOp`` referenced by the ``decomposition`` symbol
    attribute of the composite op.
    """
    if decomposition_fn is None:
        raise ValueError(
            "stablehlo_composite_golden requires `decomposition_fn` keyword "
            "(the func.FuncOp referenced by the composite's `decomposition` "
            "symbol attribute)."
        )

    if len(decomposition_fn.body.blocks) != 1:
        raise NotImplementedError(
            "stablehlo_composite_golden: multi-block decompositions are not supported."
        )
    block = decomposition_fn.body.blocks[0]
    if len(block.arguments) != len(operand_tensors):
        raise ValueError(
            "stablehlo_composite_golden: composite operand count does not match "
            "decomposition function arity."
        )

    ssa: Dict[Any, Any] = {}
    for arg_value, golden in zip(block.arguments, operand_tensors):
        ssa[arg_value] = golden

    def _resolve(value: Any) -> Any:
        if value in ssa:
            return ssa[value]
        raise NotImplementedError(
            "stablehlo_composite_golden: decomposition references a value "
            "produced outside the decomposition body (e.g. a constant or "
            "cross-block reference) which is not yet supported."
        )

    for op in block.operations:
        if isinstance(op, func.ReturnOp):
            outs = tuple(_resolve(o) for o in op.operands)
            return outs[0] if len(outs) == 1 else outs

        gfn = get_golden_function(type(op))
        operand_goldens = [_resolve(o) for o in op.operands]

        result_element_type = None
        if len(op.results) > 0:
            try:
                result_element_type = op.results[0].type.element_type
            except AttributeError:
                result_element_type = None

        try:
            out = (
                gfn(*operand_goldens, result_element_type)
                if result_element_type is not None
                else gfn(*operand_goldens)
            )
        except TypeError:
            out = gfn(*operand_goldens)

        if len(op.results) == 1:
            ssa[op.results[0]] = out
        else:
            for r, o in zip(op.results, out):
                ssa[r] = o

    raise RuntimeError(
        "stablehlo_composite_golden: decomposition function missing return."
    )


def stablehlo_reduce_golden(
    inputs: List[GoldenMapTensor],
    init_values: List[GoldenMapTensor],
    body: Region,
    dimensions: List[int],
    output_types: List[Type],
) -> GoldenMapTensor:
    if len(inputs) != len(init_values):
        raise ValueError(
            "stablehlo_reduce_golden: number of inputs must match number of init_values."
        )
    if len(inputs) != 1:
        raise NotImplementedError(
            "stablehlo_reduce_golden currently supports single-input reduces only."
        )
    if len(body.blocks) != 1:
        raise NotImplementedError(
            "stablehlo_reduce_golden: multi-block reduction bodies are not supported."
        )

    input_tensor = inputs[0]
    dims = list(dimensions)
    body_block = body.blocks[0]

    # Classify by the body op type so dispatch works regardless of source dialect.
    body_op_type = None
    for op in body_block.operations:
        if isinstance(op, (func.ReturnOp, stablehlo.ReturnOp)):
            continue
        body_op_type = type(op)
        break

    if body_op_type is None:
        raise ValueError("stablehlo_reduce_golden: reduction body has no compute op.")

    if dims:
        if body_op_type is stablehlo.AddOp:
            result = torch.sum(input_tensor, dim=dims)
        elif body_op_type is stablehlo.MaxOp:
            result = torch.amax(input_tensor, dim=dims)
        elif body_op_type is stablehlo.MinOp:
            result = torch.amin(input_tensor, dim=dims)
        elif body_op_type is stablehlo.MulOp:
            result = input_tensor
            for d in sorted(dims, reverse=True):
                result = torch.prod(result, dim=d)
        elif body_op_type is stablehlo.OrOp:
            result = torch.any(input_tensor.to(torch.bool), dim=dims).to(
                input_tensor.dtype
            )
        elif body_op_type is stablehlo.AndOp:
            result = torch.all(input_tensor.to(torch.bool), dim=dims).to(
                input_tensor.dtype
            )
        else:
            raise NotImplementedError(
                f"stablehlo_reduce_golden: unsupported body op type {body_op_type}."
            )
    else:
        result = input_tensor

    if len(output_types) >= 1:
        try:
            target_dtype = mlir_type_to_torch_dtype(output_types[0].element_type)
            result = result.to(target_dtype)
        except Exception:
            pass

    return result


def stablehlo_cbrt_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    golden_sign = torch.sign(input_tensor)
    golden_cbrt = torch.pow(torch.abs(input_tensor), 1 / 3)
    return torch.mul(golden_sign, golden_cbrt).to(output_dtype)


def stablehlo_expm1_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.expm1(input_tensor).to(output_dtype)


def stablehlo_isfinite_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.isfinite(input_tensor).to(output_dtype)


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


def stablehlo_remainder_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.remainder(input_tensor, other_tensor).to(output_dtype)


def stablehlo_atan2_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.atan2(input_tensor, other_tensor).to(output_dtype)


def stablehlo_shift_left_golden(
    input_tensor: GoldenMapTensor, other_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    input_int64 = input_tensor.to(torch.int64)
    shift_int64 = other_tensor.to(torch.int64)
    input_unsigned = torch.bitwise_and(input_int64, 0xFFFFFFFF)
    result = torch.bitwise_left_shift(input_unsigned, shift_int64)
    result = torch.bitwise_and(result, 0xFFFFFFFF)
    return result.to(output_dtype)


_STABLEHLO_COMPARE_DISPATCH = {
    "EQ": torch.eq,
    "NE": torch.ne,
    "GE": torch.ge,
    "GT": torch.gt,
    "LE": torch.le,
    "LT": torch.lt,
}


def stablehlo_compare_golden(
    lhs_tensor: GoldenMapTensor,
    rhs_tensor: GoldenMapTensor,
    direction_attr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    direction_str = str(direction_attr).upper()
    for key in _STABLEHLO_COMPARE_DISPATCH:
        if key in direction_str:
            direction_str = key
            break
    compare_fn = _STABLEHLO_COMPARE_DISPATCH.get(direction_str)
    if compare_fn is None:
        raise ValueError(f"Unsupported comparison direction: {direction_attr}")
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return compare_fn(lhs_tensor, rhs_tensor).to(output_dtype)


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


def ttnn_acos_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.acos(input_tensor).to(dtype)


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


def ttnn_asin_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.asin(input_tensor).to(dtype)


def ttnn_asinh_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.asinh(input_tensor).to(dtype)


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


# Torch goldens for fused matmul/linear activations. Mirrors
# ttnn.operations.activations._get_golden_map_for_unary_op (string keys are
# lowercase op names; ``*_approx`` suffix is stripped before lookup).
_FUSED_ACTIVATION_FNS: Dict[str, Callable] = {
    "relu": torch.nn.functional.relu,
    "relu6": torch.nn.functional.relu6,
    "silu": torch.nn.functional.silu,
    "mish": torch.nn.functional.mish,
    "sigmoid": torch.nn.functional.sigmoid,
    "hardsigmoid": torch.nn.functional.hardsigmoid,
    "tanh": torch.nn.functional.tanh,
    "log": torch.log,
    "softplus": torch.nn.functional.softplus,
    "gelu": torch.nn.functional.gelu,
    "sqrt": torch.sqrt,
}


def _get_fused_activation_fn(
    activation: Optional[Union[str, StringAttr]],
) -> Optional[Callable]:
    if activation is None:
        return None
    if not isinstance(activation, str):
        activation = unpack_mlir_attr(activation)
    name = activation[:-7] if activation.endswith("_approx") else activation
    activation_fn = _FUSED_ACTIVATION_FNS.get(name)
    if activation_fn is None:
        raise ValueError(f"Unsupported fused activation: {activation}")
    return activation_fn


def ttnn_matmul_golden(
    input_tensor: GoldenMapTensor,
    other_tensor: GoldenMapTensor,
    transpose_a_attr: BoolAttr,
    transpose_b_attr: BoolAttr,
    output_type_mlir: Type,
    activation_attr: Optional[StringAttr] = None,
) -> GoldenMapTensor:
    transpose_a = unpack_mlir_attr(transpose_a_attr)
    transpose_b = unpack_mlir_attr(transpose_b_attr)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    a = torch.transpose(input_tensor, -2, -1) if transpose_a else input_tensor
    b = torch.transpose(other_tensor, -2, -1) if transpose_b else other_tensor
    output = torch.matmul(a, b)
    activation_fn = _get_fused_activation_fn(activation_attr)
    if activation_fn is not None:
        output = activation_fn(output)
    return output.to(output_dtype)


def ttnn_linear_golden(
    input_tensor: GoldenMapTensor,
    other_tensor: GoldenMapTensor,
    bias_tensor: Optional[GoldenMapTensor],
    transpose_a_attr: BoolAttr,
    transpose_b_attr: BoolAttr,
    output_type_mlir: Type,
    activation_attr: Optional[StringAttr] = None,
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
    output = torch.add(output, bias_tensor)
    activation_fn = _get_fused_activation_fn(activation_attr)
    if activation_fn is not None:
        output = activation_fn(output)
    return output.to(output_dtype)


def ttnn_rms_norm_pre_all_gather_golden(
    input: GoldenMapTensor,
    residual: Optional[GoldenMapTensor],
    output_type_mlir: Type,
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    TILE_WIDTH = 32

    input_float = input.float()
    if residual is not None:
        input_float = input_float + residual.float()

    # Compute per-row partial statistics: E(x^2).
    # Build output per-shard to preserve GoldenMapTensor structure.
    def compute_stats(shard):
        shard_float = shard.float()
        ex2 = shard_float.square().mean(dim=-1, keepdim=True)
        output_shape = list(shard_float.shape)
        output_shape[-1] = TILE_WIDTH
        output = torch.zeros(output_shape, dtype=torch.float32)
        output[..., :1] = ex2
        return output.to(output_dtype)

    return GoldenMapTensor.apply_shardwise(input_float, compute_stats)


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
    # Upcast weight/bias to match input's float32 dtype. torch.layer_norm
    # rejects mixed fp32 input + bf16 params on CPU.
    weight_float = weight.float() if weight is not None else None
    bias_float = bias.float() if bias is not None else None

    return torch.nn.functional.layer_norm(
        input_float,
        normalized_shape=normalized_shape,
        weight=weight_float,
        bias=bias_float,
        eps=epsilon,
    ).to(output_dtype)


def ttnn_layer_norm_pre_all_gather_golden(
    input: GoldenMapTensor,
    residual_input: Optional[GoldenMapTensor],
    recip: Optional[GoldenMapTensor],
    output_type_mlir: Type,
) -> GoldenMapTensor:
    # `recip` is a precomputed reciprocal LUT [1/1, 1/2, ..., 1/width] used by
    # the Welford kernel path to replace expensive divisions with multiplies.
    # It affects *how* the device computes sum(x) and sum(x^2) (numerical stability
    # and performance), but not *what* the mathematical result is. The golden
    # reference computes the same statistics via torch.sum(), so `recip` is
    # intentionally unused here.
    del recip

    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    TILE_WIDTH = 32

    input_float = input.float()
    if residual_input is not None:
        input_float = input_float + residual_input.float()

    # Compute per-row partial statistics: sum(x^2) and sum(x).
    # The hardware kernel uses PoolType::SUM with scaler=1.0, so it outputs
    # raw sums (not means). These are combined post-all-gather.
    # Build output per-shard to preserve GoldenMapTensor structure.
    def compute_stats(shard):
        shard_float = shard.float()
        sum_x2 = shard_float.square().sum(dim=-1, keepdim=True)
        sum_x = shard_float.sum(dim=-1, keepdim=True)
        output_shape = list(shard_float.shape)
        output_shape[-1] = 2 * TILE_WIDTH
        output = torch.zeros(output_shape, dtype=torch.float32)
        output[..., :1] = sum_x2
        output[..., TILE_WIDTH : TILE_WIDTH + 1] = sum_x
        return output.to(output_dtype)

    return GoldenMapTensor.apply_shardwise(input_float, compute_stats)


def ttnn_layer_norm_post_all_gather_golden(
    input: GoldenMapTensor,
    stats: GoldenMapTensor,
    weight: Optional[GoldenMapTensor],
    bias: Optional[GoldenMapTensor],
    epsilon: FloatAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    # The stats tensor is ignored for the golden reference.  The hardware
    # kernel reconstructs E(x) and E(x^2) from the gathered statistics and
    # then applies standard layer normalization.  Rather than replicating the
    # kernel's tiled-reduce logic (which depends on tile width, device count
    # and a bfloat16 scaler), we compute the reference output directly using
    # the input tensor — exactly as tt-metal's own test does.
    del stats

    epsilon = unpack_mlir_attr(epsilon)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)

    def compute_ln(shard):
        shard_float = shard.float()
        normalized_shape = shard_float.shape[-1:]
        w = weight.shard_map[0].float() if weight is not None else None
        b = bias.shard_map[0].float() if bias is not None else None
        out = torch.nn.functional.layer_norm(
            shard_float, normalized_shape, w, b, epsilon
        )
        return out.to(output_dtype)

    return GoldenMapTensor.apply_shardwise(input, compute_ln)


def ttnn_group_norm_golden(
    input: GoldenMapTensor,
    weight: Optional[GoldenMapTensor],
    bias: Optional[GoldenMapTensor],
    num_groups,
    epsilon: FloatAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    num_groups = unpack_mlir_attr(num_groups)
    epsilon = unpack_mlir_attr(epsilon)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    input_float = input.float()

    return torch.nn.functional.group_norm(
        input_float,
        num_groups=num_groups,
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


def ttnn_full_golden(
    shape_attr: Attribute,
    fill_value_attr: Union[IntegerAttr, FloatAttr],
    mesh_shape_attr: DenseI32ArrayAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    shape = ttnn.ir.ShapeAttr.maybe_downcast(shape_attr).shape
    fill_value = unpack_mlir_attr(fill_value_attr)
    mesh_shape = unpack_mlir_attr(mesh_shape_attr)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    tensor = torch.full(shape, fill_value).to(output_dtype)
    return GoldenMapTensor(
        {i: tensor.clone() for i in range(mesh_shape[0] * mesh_shape[1])}, mesh_shape
    )


def ttnn_constant_golden(
    value_attr: DenseElementsAttr,
    mesh_shape_attr: DenseI32ArrayAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    shape = list(value_attr.type.shape)
    mesh_shape = unpack_mlir_attr(mesh_shape_attr)
    dtype = mlir_type_to_torch_dtype(value_attr.type.element_type)

    if value_attr.is_splat:
        value_attr = value_attr.get_splat_value()
        torch_tensor = torch.full(shape, value_attr.value, dtype=dtype)
    else:
        # PyTorch bfloat16 is packed as uint16 bits in DenseElementsAttr
        # MLIR's Python bindings don't support np.array() on bf16 DenseElementsAttr
        # Extract the hex-encoded data instead
        if dtype == torch.bfloat16:
            attr_str = str(value_attr)
            # MLIR uses hex encoding for large tensors: dense<"0xHEXDATA"> : tensor<...xbf16>
            match = re.search(r'"0x([0-9A-F]+)"', attr_str, re.IGNORECASE)
            if match:
                hex_str = match.group(1)
                byte_data = bytes.fromhex(hex_str)
                u16_array = np.frombuffer(byte_data, dtype=np.uint16)
                torch_tensor = (
                    torch.from_numpy(u16_array.astype(np.int16))
                    .view(torch.bfloat16)
                    .reshape(shape)
                )
            else:
                # Small tensors might use dense<[[value_attr, ...]]> format
                # Parse the float values and convert to bfloat16
                raise NotImplementedError(
                    f"Non-hex bfloat16 constant not yet supported: {attr_str[:100]}"
                )
        else:
            torch_tensor = torch.tensor(np.array(value_attr), dtype=dtype).reshape(
                shape
            )

    result = torch_tensor.reshape(shape)
    return GoldenMapTensor(
        {i: result.clone() for i in range(mesh_shape[0] * mesh_shape[1])}, mesh_shape
    )


def ttnn_reshape_golden(
    input_tensor: GoldenMapTensor, shape_attr: Attribute, output_type_mlir: Type
) -> GoldenMapTensor:
    new_shape = unpack_mlir_attr(shape_attr)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return torch.reshape(input_tensor, new_shape).to(output_dtype)


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


def _parse_mapper_config_shard_dims(mapper_config) -> list:
    """Extract shard dimensions from a MeshMapperConfig attribute.
    Positive values are shard dims, -1 means replicate.

    NOTE: This parses the string representation of the MLIR attribute via regex.
    If the Python bindings ever expose structured access to placements, prefer
    that over string parsing to avoid breakage from formatting changes.
    """
    config_str = str(mapper_config)
    placements_match = re.search(r"placements\s*=\s*\[(.*?)\]", config_str)
    if not placements_match:
        return []
    inner = placements_match.group(1)
    shard_dims = []
    for item in re.finditer(r"<([^>]+)>", inner):
        placement = item.group(1).strip()
        if placement.startswith("shard"):
            dim = int(re.search(r"(\d+)", placement.split(",")[1]).group(1))
            shard_dims.append(dim)
        else:
            shard_dims.append(-1)
    return shard_dims


def _parse_composer_config_dims(composer_config) -> list:
    """Extract compose dimensions from a MeshComposerConfig attribute.

    NOTE: This parses the string representation of the MLIR attribute via regex.
    If the Python bindings ever expose structured access to dims, prefer
    that over string parsing to avoid breakage from formatting changes.
    """
    config_str = str(composer_config)
    dims_match = re.search(r"dims\s*=\s*\[(.*?)\]", config_str)
    if dims_match:
        dims_str = dims_match.group(1)
        return [
            int(d.strip().split(":")[0].strip())
            for d in dims_str.split(",")
            if d.strip()
        ]
    return []


def ttnn_distribute_tensor_golden(
    input: GoldenMapTensor,
    mapper_config,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    mesh_shape = input.mesh_shape
    shard_dims = _parse_mapper_config_shard_dims(mapper_config)
    return apply_sharding(input, mesh_shape, shard_dims)


def ttnn_aggregate_tensor_golden(
    input: GoldenMapTensor,
    composer_config,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    mesh_shape = input.mesh_shape
    shard_dims = _parse_composer_config_dims(composer_config)
    return apply_unsharding(input, mesh_shape, shard_dims)


def ttnn_all_gather_golden(
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
        for device_id in group.keys():
            output_shards[device_id] = gathered_tensor.clone().to(output_dtype)
    return GoldenMapTensor(
        {i: t for i, t in enumerate(output_shards)}, input.mesh_shape
    )


def ttnn_gather_dim_golden(
    input_tensor: GoldenMapTensor,
    index: GoldenMapTensor,
    dim: IntegerAttr,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    dim_value = unpack_mlir_attr(dim)
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    index_copy = index.clone()
    index_copy = index_copy.to(torch.int64)
    out_tensor = torch.gather(input_tensor, dim_value, index_copy)
    return out_tensor.to(output_dtype)


def ttnn_all_reduce_async_golden(
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


def ttnn_reduce_scatter_golden(
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


################ TTNN Layout/Device Op Golden Functions ###############


def ttnn_to_layout_golden(
    input_tensor: GoldenMapTensor,
    layout_attr: Attribute,
    output_type_mlir: Type,
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    output_tensor = input_tensor.clone()
    return output_tensor.to(output_dtype)


def ttnn_to_device_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return input_tensor.clone().to(output_dtype)


def ttnn_from_device_golden(
    input_tensor: GoldenMapTensor, output_type_mlir: Type
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    return input_tensor.clone().to(output_dtype)


################ Debug Op Golden Functions ###############


def ttir_paged_flash_multi_latent_attention_decode_golden(
    query: GoldenMapTensor,
    key: GoldenMapTensor,
    value: Optional[GoldenMapTensor] = None,
    page_table: Optional[GoldenMapTensor] = None,
    attention_mask: Optional[GoldenMapTensor] = None,
    cur_pos_tensor: Optional[GoldenMapTensor] = None,
    attention_sink: Optional[GoldenMapTensor] = None,
    head_dim_v: Optional[IntegerAttr] = None,
    is_causal: Optional[BoolAttr] = None,
    scale: Optional[FloatAttr] = None,
    output_type_mlir: Optional[Type] = None,
    **kwargs,
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    head_dim_v_val = (
        unpack_mlir_attr(head_dim_v) if head_dim_v is not None else query.shape[-1]
    )
    scale_val = unpack_mlir_attr(scale) if scale is not None else 1.0
    is_causal_val = unpack_mlir_attr(is_causal) if is_causal is not None else True

    def _golden_per_shard(
        q, k, pt, v=None, cur_pos=None, attn_mask_in=None, attn_sink=None
    ):
        # Q is (S, B, H, D) from device layout, permute to (B, H, S, D).
        q = q.permute(1, 2, 0, 3).float()
        b, nh, _, d = q.shape

        # Unpage K cache: K is (num_blocks, nkv, block_size, D).
        num_blocks, nkv, block_size, _ = k.shape
        # page_table is (B, blocks_per_user), maps virtual->physical block indices.
        pt = pt.long()
        blocks_per_user = pt.shape[-1]
        seq_len = blocks_per_user * block_size

        # Gather physical blocks using page table and reshape to (B, nkv, seq_len, D).
        k_unpaged = k[pt.view(-1)]  # (B * blocks_per_user, nkv, block_size, D)
        k_unpaged = k_unpaged.reshape(b, blocks_per_user, nkv, block_size, d)
        k_unpaged = k_unpaged.transpose(
            1, 2
        )  # (B, nkv, blocks_per_user, block_size, D)
        k_unpaged = k_unpaged.reshape(b, nkv, seq_len, d).float()

        # V is derived from K's first head_dim_v dimensions if not provided.
        if v is not None:
            # Unpage V using the same page table as K.
            # V is (num_blocks, nkv, block_size, head_dim_v).
            dv = v.shape[-1]
            v_unpaged = v[pt.view(-1)]  # (B * blocks_per_user, nkv, block_size, dv)
            v_unpaged = v_unpaged.reshape(b, blocks_per_user, nkv, block_size, dv)
            v_unpaged = v_unpaged.transpose(
                1, 2
            )  # (B, nkv, blocks_per_user, block_size, dv)
            v_unpaged = v_unpaged.reshape(b, nkv, seq_len, dv).float()
        else:
            v_unpaged = k_unpaged[..., :head_dim_v_val]  # (B, nkv, seq_len, head_dim_v)

        # Expand KV heads to match Q heads (GQA expansion).
        head_rep = nh // nkv
        k_exp = k_unpaged.repeat_interleave(head_rep, dim=1)  # (B, nh, seq_len, D)
        v_exp = v_unpaged.repeat_interleave(
            head_rep, dim=1
        )  # (B, nh, seq_len, head_dim_v)

        # Build attention mask.
        attn_mask = None
        if attn_mask_in is not None:
            attn_mask = attn_mask_in.float()
        elif is_causal_val and cur_pos is not None:
            attn_mask = torch.zeros((b, nh, 1, seq_len), dtype=torch.float32)
            for i in range(b):
                start_idx = int(cur_pos[i].item())
                attn_mask[i, :, :, start_idx + 1 :] = torch.finfo(torch.float32).min

        # Apply attention sink: keep the first N positions always visible.
        if attn_sink is not None and attn_mask is not None:
            sink_len = attn_sink.shape[-1] if attn_sink.dim() > 0 else 1
            attn_mask[..., :sink_len] = 0

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k_exp, v_exp, attn_mask=attn_mask, scale=scale_val, is_causal=False
        )  # (B, nh, 1, head_dim_v)

        # Permute back to device layout (S, B, H, head_dim_v).
        out = out.permute(2, 0, 1, 3)
        return out.to(output_dtype)

    # Extract per-shard tensors and compute golden.
    q_shards = query._shard_map
    k_shards = key._shard_map
    pt_shards = page_table._shard_map if page_table is not None else {0: None}
    v_shards = value._shard_map if value is not None else {i: None for i in q_shards}
    cp_shards = (
        cur_pos_tensor._shard_map
        if cur_pos_tensor is not None
        else {i: None for i in q_shards}
    )
    am_shards = (
        attention_mask._shard_map
        if attention_mask is not None
        else {i: None for i in q_shards}
    )
    as_shards = (
        attention_sink._shard_map
        if attention_sink is not None
        else {i: None for i in q_shards}
    )

    output_shards = {}
    for shard_id in q_shards:
        output_shards[shard_id] = _golden_per_shard(
            q_shards[shard_id],
            k_shards[shard_id],
            pt_shards[shard_id],
            v=v_shards[shard_id],
            cur_pos=cp_shards[shard_id],
            attn_mask_in=am_shards[shard_id],
            attn_sink=as_shards[shard_id],
        )

    return GoldenMapTensor(output_shards, query.mesh_shape)


def ttir_sdpa_golden(
    query: GoldenMapTensor,
    key: GoldenMapTensor,
    value: GoldenMapTensor,
    attention_mask: Optional[GoldenMapTensor],
    is_causal_attr: BoolAttr,
    scale_attr: Optional[FloatAttr],
    output_type_mlir: Type,
    sliding_window_size_attr: Optional[IntegerAttr] = None,
    attention_sink: Optional[GoldenMapTensor] = None,
) -> GoldenMapTensor:
    """
    Matches tt-metal's SDPA implementation, which follows PyTorch semantics:
    softmax(QK * scale + mask). The kernel folds scale into exp, but the host
    wrapper pre-multiplies any user attn_mask by 1/scale to compensate.
    Supports standard attention and Grouped-Query Attention (GQA).

    sliding_window_size: kernel-derived {0, -inf} mask added after scaling.
      causal:     window covers last W tokens [i-W+1, i]
      non-causal: window covers [i-W/2, i+W/2] (inclusive, W+1 tokens)
    attention_sink: per-head logit treated as a virtual K column. Kernel
      applies scale to it just like raw QK, so the golden pre-scales it
      before concat-softmax-slice.
    """
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    is_causal = unpack_mlir_attr(is_causal_attr)
    scale = unpack_mlir_attr(scale_attr) if scale_attr is not None else None
    sliding_window_size = (
        unpack_mlir_attr(sliding_window_size_attr)
        if sliding_window_size_attr is not None
        else None
    )

    q_heads = query.shape[1]
    kv_heads = key.shape[1]

    if q_heads != kv_heads:
        assert q_heads % kv_heads == 0
        num_repeats = q_heads // kv_heads
        key = torch.repeat_interleave(key, num_repeats, dim=1)
        value = torch.repeat_interleave(value, num_repeats, dim=1)

    if scale is None:
        scale = 1.0 / (float(query.shape[-1]) ** 0.5)

    qk = torch.matmul(query.float(), key.float().transpose(-2, -1))
    qk = torch.mul(qk, scale)

    # When sliding_window_size is set, the kernel uses only the window mask
    # (which already encodes the causal constraint via its topology). The
    # separate causal mask path is skipped, matching the decomposition's
    # if/else-if structure.
    if sliding_window_size is not None:
        seq_len_q = qk.shape[-2]
        seq_len_k = qk.shape[-1]
        i_idx = torch.arange(seq_len_q).unsqueeze(1)
        j_idx = torch.arange(seq_len_k).unsqueeze(0)
        diff = i_idx - j_idx
        if is_causal:
            in_window = (diff >= 0) & (diff < sliding_window_size)
        else:
            half = sliding_window_size // 2
            in_window = (diff >= -half) & (diff <= half)
        window_mask = torch.where(in_window, 0.0, float("-inf"))
        qk = torch.add(qk, window_mask)
    elif is_causal and attention_mask is None:
        seq_len_q = qk.shape[-2]
        seq_len_k = qk.shape[-1]
        causal_mask = torch.triu(
            torch.full((seq_len_q, seq_len_k), float("-inf")), diagonal=1
        )
        qk = torch.add(qk, causal_mask)

    if attention_mask is not None:
        qk = torch.add(qk, attention_mask.float())

    if attention_sink is not None:
        # Sink shape: [1, Hq, 1, 1]; broadcast to [B, Hq, Sq, 1] and scale.
        # GoldenMapTensor routes torch ops through __torch_function__ but
        # doesn't override Python `*` or mutating methods like .expand — use
        # torch.* free functions instead.
        sink = torch.mul(attention_sink.float(), scale)
        sink = torch.broadcast_to(
            sink, (qk.shape[0], qk.shape[1], qk.shape[2], sink.shape[-1])
        )
        sink_cols = sink.shape[-1]
        extended = torch.cat([qk, sink], dim=-1)
        attn_weights = torch.softmax(extended, dim=-1)
        attn_weights = attn_weights[..., :-sink_cols]
    else:
        attn_weights = torch.softmax(qk, dim=-1)

    output = torch.matmul(attn_weights, value.float())

    return output.to(output_dtype)


def flash_mla_prefill_golden(
    query: GoldenMapTensor,
    key: GoldenMapTensor,
    value: Optional[GoldenMapTensor],
    attention_mask: Optional[GoldenMapTensor],
    head_dim_v: int,
    is_causal: bool,
    scale: Optional[float],
) -> GoldenMapTensor:
    """
    Golden for the tt.flash_mla_prefill custom_call.
    """
    output_dtype = query.dtype

    value_t = key[..., :head_dim_v] if value is None else value
    attn_mask = attention_mask.float() if attention_mask is not None else None
    effective_causal = is_causal and attention_mask is None

    # Compute in f32 for golden accuracy, then cast back to the query dtype.
    # A None `scale` lets SDPA apply its 1/sqrt(head_dim) default.
    output = torch.nn.functional.scaled_dot_product_attention(
        query.float(),
        key.float(),
        value_t.float(),
        attn_mask=attn_mask,
        is_causal=effective_causal,
        scale=scale,
        enable_gqa=query.shape[1] != key.shape[1],
    )

    return output.to(output_dtype)


def ttir_paged_sdpa_decode_golden(
    query: GoldenMapTensor,
    key: GoldenMapTensor,
    value: GoldenMapTensor,
    page_table: GoldenMapTensor,
    output: GoldenMapTensor,
    is_causal_attr: BoolAttr,
    attention_mask: Optional[GoldenMapTensor] = None,
    cur_pos_tensor: Optional[GoldenMapTensor] = None,
    attention_sink: Optional[GoldenMapTensor] = None,
    scale_attr: Optional[FloatAttr] = None,
    sliding_window_size_attr: Optional[IntegerAttr] = None,
    output_type_mlir: Optional[Type] = None,
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    scale_val = unpack_mlir_attr(scale_attr) if scale_attr is not None else None
    sliding_window_size_val = (
        unpack_mlir_attr(sliding_window_size_attr)
        if sliding_window_size_attr is not None
        else None
    )
    is_causal_val = unpack_mlir_attr(is_causal_attr)

    query_t = _gmt_leaf_torch(query)
    key_t = _gmt_leaf_torch(key)
    value_t = _gmt_leaf_torch(value)
    pt = _gmt_leaf_torch(page_table.long())

    # Query: [B, S, H, D] -> [B, H, S, D]
    q = query_t.float().permute(0, 2, 1, 3)
    b, nh, s_q, d = q.shape

    # K/V are paged: [num_blocks, num_kv_heads, block_size, head_dim]
    num_blocks, nkv, block_size, _ = key_t.shape
    blocks_per_user = pt.shape[-1]
    seq_len = blocks_per_user * block_size

    # Clamp indices so random golden page tables stay in-range (parse tests only).
    pt_flat = pt.view(-1).clamp(0, num_blocks - 1)

    # Unpage K using page table
    k_unpaged = key_t[pt_flat]
    k_unpaged = k_unpaged.reshape(b, blocks_per_user, nkv, block_size, d)
    k_unpaged = k_unpaged.transpose(1, 2).reshape(b, nkv, seq_len, d).float()

    # Unpage V using page table
    dv = value_t.shape[-1]
    v_unpaged = value_t[pt_flat]
    v_unpaged = v_unpaged.reshape(b, blocks_per_user, nkv, block_size, dv)
    v_unpaged = v_unpaged.transpose(1, 2).reshape(b, nkv, seq_len, dv).float()

    # GQA expansion
    head_rep = nh // nkv
    if head_rep > 1:
        k_unpaged = k_unpaged.repeat_interleave(head_rep, dim=1)
        v_unpaged = v_unpaged.repeat_interleave(head_rep, dim=1)

    # Build attention mask
    attn_mask = None
    if attention_mask is not None:
        attn_mask = _gmt_leaf_torch(attention_mask.float())
    elif is_causal_val and cur_pos_tensor is not None:
        cur_t = _gmt_leaf_torch(cur_pos_tensor)
        attn_mask = torch.zeros((b, nh, s_q, seq_len), dtype=torch.float32)
        for i in range(b):
            start_idx = int(cur_t[i].item())
            attn_mask[i, :, :, start_idx + 1 :] = torch.finfo(torch.float32).min

    # Apply sliding window mask if specified
    if sliding_window_size_val is not None and cur_pos_tensor is not None:
        cur_t = _gmt_leaf_torch(cur_pos_tensor)
        if attn_mask is None:
            attn_mask = torch.zeros((b, nh, s_q, seq_len), dtype=torch.float32)
        for i in range(b):
            start_idx = int(cur_t[i].item())
            # Mask tokens outside the sliding window
            window_start = max(0, start_idx - sliding_window_size_val + 1)
            if window_start > 0:
                attn_mask[i, :, :, :window_start] = torch.finfo(torch.float32).min

    if attention_sink is not None and attn_mask is not None:
        sink_t = _gmt_leaf_torch(attention_sink)
        sink_len = sink_t.shape[-1] if sink_t.dim() > 0 else 1
        attn_mask[..., :sink_len] = 0

    out = torch.nn.functional.scaled_dot_product_attention(
        q, k_unpaged, v_unpaged, attn_mask=attn_mask, scale=scale_val, is_causal=False
    )

    # [B, H, S, D] -> [B, S, H, D]
    out = out.permute(0, 2, 1, 3).to(output_dtype)
    return GoldenMapTensor(
        {k: out.clone() for k in query.shard_map.keys()},
        query.mesh_shape,
    )


def ttir_chunked_scaled_dot_product_attention_golden(
    query: GoldenMapTensor,
    key: GoldenMapTensor,
    value: GoldenMapTensor,
    page_table: GoldenMapTensor,
    chunk_start_idx: GoldenMapTensor,
    output: GoldenMapTensor,
    scale_attr: Optional[FloatAttr] = None,
    output_type_mlir: Optional[Type] = None,
) -> GoldenMapTensor:
    output_dtype = mlir_type_to_torch_dtype(output_type_mlir)
    scale_val = unpack_mlir_attr(scale_attr) if scale_attr is not None else None

    query_t = _gmt_leaf_torch(query)
    key_t = _gmt_leaf_torch(key)
    value_t = _gmt_leaf_torch(value)
    pt = _gmt_leaf_torch(page_table.long())
    start_t = _gmt_leaf_torch(chunk_start_idx.long())

    # Query is already [B, H, S, D] for chunked prefill (no permute).
    q = query_t.float()
    b, nh, s_q, d = q.shape

    # K/V are paged: [num_blocks, num_kv_heads, block_size, head_dim].
    num_blocks, nkv, block_size, _ = key_t.shape
    blocks_per_user = pt.shape[-1]
    seq_len = blocks_per_user * block_size

    # Clamp indices so random golden page tables stay in-range (parse tests only).
    pt_flat = pt.view(-1).clamp(0, num_blocks - 1)

    # Unpage K using page table -> [B, num_kv_heads, seq_len, D].
    k_unpaged = key_t[pt_flat]
    k_unpaged = k_unpaged.reshape(b, blocks_per_user, nkv, block_size, d)
    k_unpaged = k_unpaged.transpose(1, 2).reshape(b, nkv, seq_len, d).float()

    # Unpage V using page table.
    dv = value_t.shape[-1]
    v_unpaged = value_t[pt_flat]
    v_unpaged = v_unpaged.reshape(b, blocks_per_user, nkv, block_size, dv)
    v_unpaged = v_unpaged.transpose(1, 2).reshape(b, nkv, seq_len, dv).float()

    # GQA expansion.
    head_rep = nh // nkv
    if head_rep > 1:
        k_unpaged = k_unpaged.repeat_interleave(head_rep, dim=1)
        v_unpaged = v_unpaged.repeat_interleave(head_rep, dim=1)

    # Causal mask: query row j (absolute position start + j) attends to key
    # columns [0, start + j] inclusive; everything else is masked.
    start = int(start_t.view(-1)[0].item())
    col = torch.arange(seq_len).view(1, seq_len)
    row = (start + torch.arange(s_q)).view(s_q, 1)
    allowed = col <= row
    attn_mask = torch.zeros((1, 1, s_q, seq_len), dtype=torch.float32)
    attn_mask.masked_fill_(
        ~allowed.view(1, 1, s_q, seq_len), torch.finfo(torch.float32).min
    )

    out = torch.nn.functional.scaled_dot_product_attention(
        q, k_unpaged, v_unpaged, attn_mask=attn_mask, scale=scale_val, is_causal=False
    )

    # Output mirrors query shape [B, H, S, D].
    out = out.to(output_dtype)
    return GoldenMapTensor(
        {k: out.clone() for k in query.shard_map.keys()},
        query.mesh_shape,
    )


def _gmt_leaf_torch(t: Union[GoldenMapTensor, torch.Tensor]) -> torch.Tensor:
    """Resolve GoldenMapTensor (recursively) to a torch.Tensor for scalar/index ops."""
    while isinstance(t, GoldenMapTensor):
        keys = sorted(t.shard_map.keys())
        t = t.shard_map[keys[0]]
    return t


def ttir_paged_update_cache_golden(
    cache_tensor: GoldenMapTensor,
    input_tensor: GoldenMapTensor,
    update_index_tensor: GoldenMapTensor,
    share_cache_attr: BoolAttr,
    page_table_tensor: Optional[GoldenMapTensor] = None,
    output_type_mlir: Optional[Type] = None,
) -> GoldenMapTensor:
    result = cache_tensor.clone()
    # cache: [num_blocks, num_heads, block_size, head_dim]
    # input: [batch, seq_len, num_heads, head_dim]
    # update_index: [batch] - sequence position to update
    # page_table: [batch, max_blocks_per_seq]
    block_size = cache_tensor.shape[2]
    indices = _gmt_leaf_torch(update_index_tensor.to(torch.long))
    batch = input_tensor.shape[0]
    seq_len = input_tensor.shape[1]
    page_t = (
        _gmt_leaf_torch(page_table_tensor) if page_table_tensor is not None else None
    )
    num_blocks = cache_tensor.shape[0]

    for device_id, res_shard in result.shard_map.items():
        inp_shard = input_tensor.shard_map[device_id]
        for b_idx in range(batch):
            for s in range(seq_len):
                pos = indices[b_idx].item() + s
                block_idx = pos // block_size
                offset = pos % block_size
                if page_t is not None:
                    pg_cols = page_t.size(1)
                    pg_idx = min(max(block_idx, 0), pg_cols - 1)
                    physical_block = page_t[b_idx, pg_idx].long().item()
                else:
                    physical_block = block_idx
                physical_block = min(max(int(physical_block), 0), num_blocks - 1)
                res_shard[:, :, offset, :][physical_block] = inp_shard[b_idx, s, :, :]
    return result


def ttir_paged_fill_cache_golden(
    cache_tensor: GoldenMapTensor,
    input_tensor: GoldenMapTensor,
    page_table_tensor: GoldenMapTensor,
    batch_idx_tensor: Optional[GoldenMapTensor] = None,
    output_type_mlir: Optional[Type] = None,
) -> GoldenMapTensor:
    result = cache_tensor.clone()
    # cache: [num_blocks, num_heads, block_size, head_dim]
    # input: [batch, num_heads, seq_len, head_dim]
    # page_table: [batch, max_blocks_per_seq]
    block_size = cache_tensor.shape[2]
    batch = input_tensor.shape[0]
    seq_len = input_tensor.shape[2]
    num_blocks = cache_tensor.shape[0]
    page_t = _gmt_leaf_torch(page_table_tensor)
    max_pg_rows = page_t.size(0)
    max_pg_cols = page_t.size(1)
    batch_indices_t = (
        _gmt_leaf_torch(batch_idx_tensor.to(torch.long).reshape(-1))
        if batch_idx_tensor is not None
        else None
    )

    for device_id, res_shard in result.shard_map.items():
        inp_shard = input_tensor.shard_map[device_id]
        for b_idx in range(batch):
            page_table_batch_idx = (
                int(batch_indices_t[b_idx].item())
                if batch_indices_t is not None
                else b_idx
            )
            page_table_batch_idx = min(max(page_table_batch_idx, 0), max_pg_rows - 1)
            for seq_pos in range(seq_len):
                blk_idx = seq_pos // block_size
                offset = seq_pos % block_size
                pg_col = min(max(blk_idx, 0), max_pg_cols - 1)
                physical_block = int(page_t[page_table_batch_idx, pg_col].long().item())
                physical_block = min(max(physical_block, 0), num_blocks - 1)
                res_shard[physical_block, :, offset, :] = inp_shard[
                    b_idx, :, seq_pos, :
                ]
    return result


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
    ttir.AcosOp: ttir_acos_golden,
    ttir.ErfOp: ttir_erf_golden,
    ttir.ErfcOp: torch.erfc,
    ttir.FloorOp: ttir_floor_golden,
    ttir.GeluOp: ttir_gelu_golden,
    ttir.GeluBackwardOp: ttir_gelu_backward_golden,
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
    ttir.AsinOp: ttir_asin_golden,
    ttir.AsinhOp: ttir_asinh_golden,
    ttir.SqrtOp: ttir_sqrt_golden,
    ttir.SquareOp: ttir_square_golden,
    ttir.LogOp: ttir_log_golden,
    ttir.Log1pOp: ttir_log1p_golden,
    ttir.Expm1Op: torch.expm1,
    ttir.ExpOp: ttir_exp_golden,
    ttir.Exp2Op: ttir_exp2_golden,
    ttir.SoftsignOp: ttir_softsign_golden,
    ttir.SignbitOp: ttir_signbit_golden,
    ttir.SeluOp: ttir_selu_golden,
    ttir.FracOp: ttir_frac_golden,
    ttir.TruncOp: ttir_trunc_golden,
    # Elementwise binary operations
    ttir.AddOp: ttir_add_golden,
    ttir.Atan2Op: ttir_atan2_golden,
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
    ttir.LogicalLeftShiftOp: ttir_logical_left_shift_golden,
    ttir.LogicalOrOp: ttir_logical_or_golden,
    ttir.LogicalRightShiftOp: ttir_logical_right_shift_golden,
    ttir.LogicalXorOp: logical_xor_golden,
    ttir.LogicalNotOp: ttir_logical_not_golden,
    ttir.RightShiftOp: ttir_right_shift_golden,
    # Selection operations
    ttir.WhereOp: ttir_where_golden,
    # Bitwise operations
    ttir.BitwiseAndOp: ttir_bitwise_and_golden,
    ttir.BitwiseOrOp: ttir_bitwise_or_golden,
    ttir.BitwiseXorOp: ttir_bitwise_xor_golden,
    ttir.BitwiseNotOp: ttir_bitwise_not_golden,
    # Reduction operations
    ttir.SumOp: ttir_sum_golden,
    ttir.MeanOp: mean_golden,
    ttir.MaxOp: ttir_max_golden,
    ttir.MinOp: min_golden,
    ttir.ProdOp: prod_golden,
    ttir.ReduceAndOp: ttir_reduce_and_golden,
    ttir.ReduceOrOp: ttir_reduce_or_golden,
    ttir.TopKOp: ttir_topk_golden,
    ttir.TopKRouterGptOp: ttir_topk_router_gpt_golden,
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
    ttir.ClampScalarOp: ttir_clamp_scalar_golden,
    ttir.ClampTensorOp: ttir_clamp_tensor_golden,
    ttir.CumSumOp: ttir_cumsum_golden,
    ttir.CumProdOp: ttir_cumprod_golden,
    ttir.BroadcastOp: ttir_broadcast_golden,
    ttir.PadOp: ttir_pad_golden,
    ttir.IndexSelectOp: select_golden,
    ttir.IndexOp: index_golden,
    ttir.SliceStaticOp: ttir_slice_golden,
    # Neural network operations
    ttir.SoftmaxOp: softmax_golden,
    ttir.MatmulOp: ttir_matmul_golden,
    ttir.EmbeddingOp: ttir_embedding_golden,
    ttir.EmbeddingBackwardOp: ttir_embedding_backward_golden,
    ttir.Upsample2dOp: upsample2d_golden,
    ttir.BatchNormInferenceOp: ttir_batch_norm_inference_golden,
    ttir.BatchNormTrainingOp: ttir_batch_norm_training_golden,
    ttir.LayerNormOp: ttir_layer_norm_golden,
    ttir.SplitQueryKeyValueAndSplitHeadsOp: ttir_split_query_key_value_and_split_heads_golden,
    ttir.GroupNormOp: ttir_group_norm_golden,
    ttir.RMSNormOp: ttir_rms_norm_golden,
    ttir.DistributedRMSNormOp: ttir_distributed_rms_norm_golden,
    ttir.DistributedLayerNormOp: ttir_distributed_layer_norm_golden,
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
    ttir.Conv3dOp: conv3d_golden,
    ttir.ConvTranspose2dOp: conv_transpose2d_golden,
    ttir.MaxPool2dOp: ttir_max_pool2d_golden,
    ttir.AvgPool2dOp: avg_pool2d_golden,
    ttir.GlobalAvgPool2dOp: global_avg_pool2d_golden,
    ttir.MaxPool2dWithIndicesOp: ttir_max_pool2d_with_indices,
    ttir.ArgMaxOp: ttir_argmax_golden,
    ttir.LinearOp: linear_golden,
    ttir.ScaledDotProductAttentionOp: ttir_sdpa_golden,
    ttir.ScaledDotProductAttentionDecodeOp: sdpa_decode_golden,
    ttir.DotGeneralOp: ttir_dot_general_golden,
    ttir.ScatterOp: ttir_scatter_golden,
    ttir.GatherOp: ttir_gather_golden,
    # Layout operations (identity functions) — accept and ignore extra kwargs like reinterpretLayout
    ttir.ToLayoutOp: ttir_to_layout_golden,
    # Cache operations
    ttir.FillCacheOp: fill_cache_golden,
    ttir.UpdateCacheOp: update_cache_golden,
    ttir.PagedUpdateCacheOp: ttir_paged_update_cache_golden,
    ttir.PagedFillCacheOp: ttir_paged_fill_cache_golden,
    # CCL (Collective Communication Library) operations
    ttir.MeshShardOp: ttir_mesh_shard_golden,
    ttir.AllGatherOp: ttir_all_gather_golden,
    ttir.AllReduceOp: ttir_all_reduce_golden,
    ttir.AllReduceAsyncOp: ttir_all_reduce_golden,
    ttir.ReduceScatterOp: ttir_reduce_scatter_golden,
    ttir.MeshPartitionOp: ttir_mesh_partition_golden,
    ttir.CollectivePermuteOp: ttir_collective_permute_golden,
    ttir.AllToAllOp: ttir_all_to_all_golden,
    ttir.CollectiveBroadcastOp: ttir_collective_broadcast_golden,
    # Sparse MoE operations
    ttir.SparseMatmulOp: ttir_sparse_matmul_golden,
    ttir.AllToAllDispatchOp: ttir_all_to_all_dispatch_golden,
    ttir.AllToAllDispatchMetadataOp: ttir_all_to_all_dispatch_metadata_golden,
    ttir.AllToAllCombineOp: ttir_all_to_all_combine_golden,
    ttir.MoeGptOp: moe_gpt_golden,
    ttir.MoeExpertTokenRemapOp: ttir_moe_expert_token_remap_golden,
    ttir.MoeComputeOp: ttir_moe_compute_golden,
    # Operations with parameter transformations
    ttir.LeakyReluOp: leaky_relu_golden,
    # Attention operations
    ttir.PagedFlashMultiLatentAttentionDecodeOp: ttir_paged_flash_multi_latent_attention_decode_golden,
    ttir.PagedScaledDotProductAttentionDecodeOp: ttir_paged_sdpa_decode_golden,
    ttir.ChunkedScaledDotProductAttentionOp: ttir_chunked_scaled_dot_product_attention_golden,
    ttir.SamplingOp: ttir_sampling_golden,
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
    stablehlo.SignOp: stablehlo_sign_golden,
    stablehlo.ConvertOp: stablehlo_convert_golden,
    stablehlo.CompositeOp: stablehlo_composite_golden,
    stablehlo.CbrtOp: stablehlo_cbrt_golden,
    stablehlo.Expm1Op: stablehlo_expm1_golden,
    stablehlo.IsFiniteOp: stablehlo_isfinite_golden,
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
    stablehlo.BroadcastInDimOp: stablehlo_broadcast_in_dim_golden,
    stablehlo.SubtractOp: stablehlo_subtract_golden,
    stablehlo.PowOp: stablehlo_pow_golden,
    stablehlo.ShiftRightLogicalOp: stablehlo_shift_right_logical_golden,
    stablehlo.RemOp: stablehlo_remainder_golden,
    stablehlo.Atan2Op: stablehlo_atan2_golden,
    stablehlo.ShiftLeftOp: stablehlo_shift_left_golden,
    stablehlo.ReverseOp: stablehlo_reverse_golden,
    stablehlo.DotGeneralOp: stablehlo_dot_general_golden,
    stablehlo.DynamicSliceOp: dynamic_slice_golden,
    stablehlo.DynamicUpdateSliceOp: stablehlo_dynamic_update_slice_golden,
    stablehlo.ConvolutionOp: stablehlo_convolution_golden,
    stablehlo.SortOp: stablehlo_sort_golden,
    stablehlo.CompareOp: stablehlo_compare_golden,
    # StableHLO tensor manipulation operations
    stablehlo.TransposeOp: stablehlo_transpose_golden,
    stablehlo.SelectOp: stablehlo_select_golden,
    stablehlo.PadOp: stablehlo_pad_golden,
    stablehlo.GatherOp: gather_golden,
    stablehlo.ScatterOp: stablehlo_scatter_golden,
    # CCL (Collective Communication Library) operations
    stablehlo.AllGatherOp: stablehlo_all_gather_golden,
    stablehlo.AllReduceOp: stablehlo_all_reduce_golden,
    stablehlo.ReduceScatterOp: stablehlo_reduce_scatter_golden,
    stablehlo.ReduceOp: stablehlo_reduce_golden,
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
    ttnn.AcosOp: ttnn_acos_golden,
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
    ttnn.AsinOp: ttnn_asin_golden,
    ttnn.AsinhOp: ttnn_asinh_golden,
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
    ttnn.LayerNormPreAllGatherOp: ttnn_layer_norm_pre_all_gather_golden,
    ttnn.LayerNormPostAllGatherOp: ttnn_layer_norm_post_all_gather_golden,
    ttnn.GroupNormOp: ttnn_group_norm_golden,
    ttnn.RMSNormOp: rms_norm_golden,
    ttnn.PagedFlashMultiLatentAttentionDecodeOp: ttir_paged_flash_multi_latent_attention_decode_golden,
    ttnn.RMSNormPreAllGatherOp: ttnn_rms_norm_pre_all_gather_golden,
    # Tensor manipulation
    ttnn.ConcatOp: ttnn_concat_golden,
    ttnn.RepeatOp: ttnn_repeat_golden,
    ttnn.RepeatInterleaveOp: ttnn_repeat_interleave_golden,
    ttnn.ClampScalarOp: ttnn_clamp_scalar_golden,
    ttnn.ClampTensorOp: ttnn_clamp_tensor_golden,
    ttnn.ReshapeOp: ttnn_reshape_golden,
    # Tensor creation
    ttnn.FullOp: ttnn_full_golden,
    ttnn.ConstantOp: ttnn_constant_golden,
    # Layout/Device operations
    ttnn.ToLayoutOp: ttnn_to_layout_golden,
    ttnn.ToDeviceOp: ttnn_to_device_golden,
    ttnn.FromDeviceOp: ttnn_from_device_golden,
    # CCL (Collective Communication Library) operations
    ttnn.DistributeTensorOp: ttnn_distribute_tensor_golden,
    ttnn.AggregateTensorOp: ttnn_aggregate_tensor_golden,
    ttnn.AllGatherOp: ttnn_all_gather_golden,
    ttnn.GatherOp: ttnn_gather_dim_golden,
    ttnn.SamplingOp: ttnn_sampling_golden,
    ttnn.AllReduceAsyncOp: ttnn_all_reduce_async_golden,
    ttnn.ReduceScatterOp: ttnn_reduce_scatter_golden,
    ttnn.MoeExpertTokenRemapOp: ttir_moe_expert_token_remap_golden,
    ttnn.MoeComputeOp: ttir_moe_compute_golden,
    # ----- DEBUG OPS -----
    debug.AnnotateOp: debug_annotate_golden,
    debug.RegionStartOp: debug_region_start_golden,
    debug.RegionEndOp: debug_region_end_golden,
}


# StableHLO custom_call goldens
STABLEHLO_CUSTOM_CALL_GOLDEN_MAPPINGS: Dict[str, Callable] = {
    "tt.flash_mla_prefill": flash_mla_prefill_golden,
}


def get_custom_call_golden_function(call_target_name: str) -> Optional[Callable]:
    """
    Get the golden function for a given stablehlo.custom_call operation name.
    """

    if call_target_name in STABLEHLO_CUSTOM_CALL_GOLDEN_MAPPINGS:
        return STABLEHLO_CUSTOM_CALL_GOLDEN_MAPPINGS[call_target_name]
    assert (
        False
    ), f"No golden function found for custom_call operation: {call_target_name}"


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
    if (ttir_op_class == ttir.ToLayoutOp) and "tilize" in kwargs:
        if kwargs["tilize"]:
            return tilize_golden
        else:
            return untilize_golden

    if ttir_op_class in GOLDEN_MAPPINGS:
        return GOLDEN_MAPPINGS[ttir_op_class]

    assert False, f"No golden function found for TTIR operation: {ttir_op_class}"


# Chisel golden interface: fn(op, inputs: Dict[str, GoldenMapTensor]) -> GoldenMapTensor | tuple
# TODO(ndrakulic, #8399) this needs unification with other goldens


def _ttnn_unflatten_nhwc(tensor, batch_size, input_height, input_width, channels):
    """Unflatten TTNN's [1, 1, N*H*W, C] layout to NHWC."""
    return tensor.reshape(batch_size, input_height, input_width, channels)


def _ttnn_flatten_nhwc(tensor):
    """Flatten an NHWC tensor back to TTNN's [1, 1, N*H*W, C] layout."""
    n, h, w, c = tensor.shape
    return tensor.reshape(1, 1, n * h * w, c)


def _chisel_unary(golden_fn):
    def wrapper(op, inputs):
        return golden_fn(
            input_tensor=inputs["input"],
            output_type_mlir=op.results[0].type.element_type,
        )

    return wrapper


def _chisel_binary(golden_fn, *, rhs_kwarg="other_tensor"):
    def wrapper(op, inputs):
        return golden_fn(
            input_tensor=inputs["lhs"],
            output_type_mlir=op.results[0].type.element_type,
            **{rhs_kwarg: inputs["rhs"]},
        )

    return wrapper


def chisel_ttnn_where(op, inputs):
    return ttnn_where_golden(
        condition=inputs["first"].bool(),
        x=inputs["second"],
        y=inputs["third"],
        output_type_mlir=op.results[0].type.element_type,
    )


def chisel_ttnn_clamp_tensor(op, inputs):
    return ttnn_clamp_tensor_golden(
        input_tensor=inputs["input"],
        min_tensor=inputs["min"],
        max_tensor=inputs["max"],
        output_type_mlir=op.results[0].type.element_type,
    )


def chisel_ttnn_typecast(op, inputs):
    return ttnn_typecast_golden(
        input_tensor=inputs["input"], output_type_mlir=op.results[0].type.element_type
    )


def chisel_ttnn_to_layout(op, inputs):
    return ttnn_to_layout_golden(
        input_tensor=inputs["input"],
        layout_attr=op.attributes["layout"],
        output_type_mlir=op.results[0].type.element_type,
    )


def chisel_ttnn_leaky_relu(op, inputs):
    return ttnn_leaky_relu_golden(
        input_tensor=inputs["input"],
        parameter_attr=op.attributes["parameter"],
        output_type_mlir=op.results[0].type.element_type,
    )


def chisel_ttnn_clamp_scalar(op, inputs):
    return ttnn_clamp_scalar_golden(
        input_tensor=inputs["input"],
        min_attr=op.attributes["min"],
        max_attr=op.attributes["max"],
        output_type_mlir=op.results[0].type.element_type,
    )


def chisel_ttnn_repeat_interleave(op, inputs):
    return ttnn_repeat_interleave_golden(
        input_tensor=inputs["input"],
        repeats_attr=op.attributes["repeats"],
        dim_attr=op.attributes["dim"],
        output_type_mlir=op.results[0].type.element_type,
    )


def chisel_ttnn_matmul(op, inputs):
    return ttnn_matmul_golden(
        input_tensor=inputs["a"],
        other_tensor=inputs["b"],
        transpose_a_attr=op.attributes["transpose_a"],
        transpose_b_attr=op.attributes["transpose_b"],
        activation_attr=_attr_get(op.attributes, "activation"),
        output_type_mlir=op.results[0].type.element_type,
    )


def chisel_ttnn_concat(op, inputs):
    return ttnn_concat_golden(
        input_tensors=tuple(inputs["inputs"]),
        dim_attr=op.attributes["dim"],
        output_type_mlir=op.results[0].type.element_type,
    )


def chisel_ttnn_repeat(op, inputs):
    return ttnn_repeat_golden(
        input_tensor=inputs["input"],
        repeat_dims_attr=op.attributes["repeat_dims"],
        output_type_mlir=op.results[0].type.element_type,
    )


def chisel_debug_annotate(op, inputs):
    return debug_annotate_golden(inputs["input"], op.attributes["annotation"])


def chisel_debug_region_start(op, inputs):
    return debug_region_start_golden(inputs["input"], op.attributes["region_id"])


def chisel_debug_region_end(op, inputs):
    return debug_region_end_golden(inputs["input"], op.attributes["region_id"])


def chisel_ttnn_assign(op, inputs):
    return (
        inputs["input"]
        .clone()
        .to(mlir_type_to_torch_dtype(op.results[0].type.element_type))
    )


def chisel_ttnn_to_memory_config(op, inputs):
    return (
        inputs["input"]
        .clone()
        .to(mlir_type_to_torch_dtype(op.results[0].type.element_type))
    )


def chisel_ttnn_deallocate(op, inputs):
    return ()


def chisel_ttnn_linear(op, inputs):
    return ttnn_linear_golden(
        input_tensor=inputs["a"],
        other_tensor=inputs["b"],
        bias_tensor=inputs["bias"],
        transpose_a_attr=op.attributes["transpose_a"],
        transpose_b_attr=op.attributes["transpose_b"],
        activation_attr=_attr_get(op.attributes, "activation"),
        output_type_mlir=op.results[0].type.element_type,
    )


def chisel_ttnn_layer_norm(op, inputs):
    # ttnn.layer_norm always normalizes over the last dimension of the input
    # (the op carries no `normalized_shape` attribute, unlike its TTIR counterpart).
    input_shape = list(op.operands[0].type.shape)
    return ttnn_layer_norm_golden(
        input=inputs["input"],
        weight=inputs["weight"],
        bias=inputs["bias"],
        normalized_shape=[input_shape[-1]],
        epsilon=op.attributes["epsilon"],
        output_type_mlir=op.results[0].type.element_type,
    )


def chisel_ttnn_layer_norm_pre_all_gather(op, inputs):
    return ttnn_layer_norm_pre_all_gather_golden(
        input=inputs["input"],
        residual_input=inputs["residual_input"],
        recip=inputs["recip"],
        output_type_mlir=op.results[0].type.element_type,
    )


def chisel_ttnn_layer_norm_post_all_gather(op, inputs):
    return ttnn_layer_norm_post_all_gather_golden(
        input=inputs["input"],
        stats=inputs["stats"],
        weight=inputs["weight"],
        bias=inputs["bias"],
        epsilon=op.attributes["epsilon"],
        output_type_mlir=op.results[0].type.element_type,
    )


def chisel_ttnn_rms_norm(op, inputs):
    input_tensor = inputs["input"]
    normalized_shape = [input_tensor.shape[-1]]
    return rms_norm_golden(
        input=input_tensor,
        weight=inputs["weight"],
        bias=inputs["bias"],
        normalized_shape=normalized_shape,
        epsilon=unpack_mlir_attr(op.attributes["epsilon"]),
    )


def chisel_ttnn_distribute_tensor(op, inputs):
    return ttnn_distribute_tensor_golden(
        input=inputs["input"],
        mapper_config=op.attributes["mapper_config"],
        output_type_mlir=op.results[0].type.element_type,
    )


def chisel_ttnn_aggregate_tensor(op, inputs):
    return ttnn_aggregate_tensor_golden(
        input=inputs["input"],
        composer_config=op.attributes["composer_config"],
        output_type_mlir=op.results[0].type.element_type,
    )


def chisel_ttnn_all_gather(op, inputs):
    return ttnn_all_gather_golden(
        input=inputs["input"],
        all_gather_dim_attr=op.attributes["all_gather_dim"],
        cluster_axis_attr=op.attributes["cluster_axis"],
        output_type_mlir=op.results[0].type.element_type,
    )


def chisel_ttnn_reduce_scatter(op, inputs):
    return ttnn_reduce_scatter_golden(
        input=inputs["input"],
        reduce_type_attr=op.attributes["reduce_type"],
        scatter_dim_attr=op.attributes["scatter_dim"],
        cluster_axis_attr=op.attributes["cluster_axis"],
        output_type_mlir=op.results[0].type.element_type,
    )


def chisel_ttnn_all_reduce_async(op, inputs):
    return ttir_all_reduce_golden(
        input=inputs["input"],
        reduce_type_attr=op.attributes["reduce_type"],
        cluster_axis_attr=op.attributes["cluster_axis"],
        output_type_mlir=op.results[0].type.element_type,
    )


def chisel_ttnn_gather(op, inputs):
    return ttnn_gather_dim_golden(
        input_tensor=inputs["input"],
        index=inputs["index"],
        dim=op.attributes["dim"],
        output_type_mlir=op.results[0].type.element_type,
    )


def chisel_ttnn_paged_flash_multi_latent_attention_decode(op, inputs):
    return ttir_paged_flash_multi_latent_attention_decode_golden(
        query=inputs["query"],
        key=inputs["key"],
        value=inputs["value"],
        page_table=inputs["page_table"],
        attention_mask=inputs["attention_mask"],
        cur_pos_tensor=inputs["cur_pos_tensor"],
        attention_sink=inputs["attention_sink"],
        head_dim_v=op.attributes["head_dim_v"],
        is_causal=_attr_get(op.attributes, "is_causal"),
        scale=_attr_get(op.attributes, "scale"),
        output_type_mlir=op.results[0].type.element_type,
    )


def chisel_ttnn_sum(op, inputs):
    return ttir_sum_golden(
        input_tensor=inputs["input"],
        dim_arg_attr=op.attributes["dim_arg"],
        keep_dim_attr=op.attributes["keep_dim"],
        output_type_mlir=op.results[0].type.element_type,
    )


def chisel_ttnn_max(op, inputs):
    return ttir_max_golden(
        input_tensor=inputs["input"],
        dim_arg_attr=op.attributes["dim_arg"],
        keep_dim_attr=op.attributes["keep_dim"],
        output_type_mlir=op.results[0].type.element_type,
    )


def chisel_ttnn_mean(op, inputs):
    return mean_golden(
        input_tensor=inputs["input"],
        dim_arg=_attr_get_value(op.attributes, "dim_arg"),
        keep_dim=unpack_mlir_attr(op.attributes["keep_dim"]),
    )


def chisel_ttnn_min(op, inputs):
    return min_golden(
        input_tensor=inputs["input"],
        dim_arg=_attr_get_value(op.attributes, "dim_arg"),
        keep_dim=unpack_mlir_attr(op.attributes["keep_dim"]),
    )


def chisel_ttnn_prod(op, inputs):
    dim_arg_val = _attr_get_value(op.attributes, "dim_arg")
    if isinstance(dim_arg_val, int):
        dim_arg_val = [dim_arg_val]
    return prod_golden(
        input_tensor=inputs["input"],
        dim_arg=dim_arg_val,
        keep_dim=unpack_mlir_attr(op.attributes["keep_dim"]),
    )


def chisel_ttnn_argmax(op, inputs):
    input_tensor = inputs["input"]
    keep_dim = unpack_mlir_attr(op.attributes["keep_dim"])
    output_dtype = mlir_type_to_torch_dtype(op.results[0].type.element_type)
    dim = _attr_get_value(op.attributes, "dim")
    if dim is not None:
        result = torch.argmax(input_tensor, dim=dim, keepdim=keep_dim)
    else:
        result = torch.argmax(input_tensor, keepdim=keep_dim)
    return result.to(output_dtype)


def chisel_ttnn_cumsum(op, inputs):
    return ttir_cumsum_golden(
        input_tensor=inputs["input"],
        dim=op.attributes["dim"],
        output_type_mlir=op.results[0].type.element_type,
    )


def chisel_ttnn_sort(op, inputs):
    values, indices = ttir_sort_golden(
        input_tensor=inputs["input"],
        dim_attr=op.attributes["dim"],
        descending_attr=op.attributes["descending"],
        stable_attr=op.attributes["stable"],
        output_type_mlir=op.results[0].type.element_type,
    )
    indices_dtype = mlir_type_to_torch_dtype(op.results[1].type.element_type)
    return values, indices.to(indices_dtype)


def chisel_ttnn_transpose(op, inputs):
    return transpose_golden(
        input_tensor=inputs["input"],
        dim0=unpack_mlir_attr(op.attributes["dim0"]),
        dim1=unpack_mlir_attr(op.attributes["dim1"]),
    )


def chisel_ttnn_reshape(op, inputs):
    return reshape_golden(
        input_tensor=inputs["input"], shape=unpack_mlir_attr(op.attributes["shape"])
    )


def chisel_ttnn_permute(op, inputs):
    return ttir_permute_golden(
        input_tensor=inputs["input"],
        permutation_attr=op.attributes["permutation"],
        output_type_mlir=op.results[0].type.element_type,
    )


def chisel_ttnn_slice_static(op, inputs):
    return ttir_slice_golden(
        input_tensor=inputs["input"],
        begins=op.attributes["begins"],
        ends=op.attributes["ends"],
        step=op.attributes["step"],
        output_type_mlir=op.results[0].type.element_type,
    )


def chisel_ttnn_slice_dynamic(op, inputs):
    return dynamic_slice_golden(
        input_tensor=inputs["input"],
        begins=inputs["begins"],
        ends=inputs["ends"],
        step=_attr_get_value(op.attributes, "step"),
    )


def chisel_ttnn_topk(op, inputs):
    values, indices = ttir_topk_golden(
        input_tensor=inputs["input_tensor"],
        k_attr=op.attributes["k"],
        dim_attr=op.attributes["dim"],
        largest_attr=op.attributes["largest"],
        sorted_attr=op.attributes["sorted"],
        output_type_mlir=op.results[0].type.element_type,
    )
    indices_dtype = mlir_type_to_torch_dtype(op.results[1].type.element_type)
    return values, indices.to(indices_dtype)


def chisel_ttnn_pad(op, inputs):
    return ttir_pad_golden(
        input_tensor=inputs["input"],
        padding=op.attributes["padding"],
        value=op.attributes["value"],
        output_type_mlir=op.results[0].type.element_type,
    )


def chisel_ttnn_softmax(op, inputs):
    return softmax_golden(
        input_tensor=inputs["input"],
        dimension=unpack_mlir_attr(op.attributes["dimension"]),
    )


def chisel_ttnn_hardsigmoid(op, inputs):
    return ttir_hardsigmoid_golden(
        input_tensor=inputs["input"], output_type_mlir=op.results[0].type.element_type
    )


def chisel_ttnn_embedding(op, inputs):
    return ttir_embedding_golden(
        indices_tensor=inputs["input"],
        weight_tensor=inputs["weight"],
        output_type_mlir=op.results[0].type.element_type,
    )


def chisel_ttnn_embedding_backward(op, inputs):
    return ttir_embedding_backward_golden(
        indices_tensor=inputs["input"],
        weight_tensor=inputs["weight"],
        in_gradient_tensor=inputs["in_gradient"],
        output_type_mlir=op.results[0].type.element_type,
    )


def chisel_ttnn_gelu_backward(op, inputs):
    return ttir_gelu_backward_golden(
        grad=inputs["lhs"],
        input=inputs["rhs"],
        approximate=_attr_get_value(op.attributes, "approximate", default="none"),
    )


def chisel_ttnn_dropout(op, inputs):
    return ttir_dropout_golden(
        input_tensor=inputs["input"],
        prob=op.attributes["prob"],
        scale=op.attributes["scale"],
        seed=op.attributes["seed"],
        use_per_device_seed=op.attributes["use_per_device_seed"],
        output_type_mlir=op.results[0].type.element_type,
    )


def chisel_ttnn_global_avg_pool2d(op, inputs):
    return global_avg_pool2d_golden(
        input_tensor=inputs["input"], output_type_mlir=op.results[0].type.element_type
    )


def chisel_ttnn_max_pool2d(op, inputs):
    # TTNN max_pool2d receives a flat [1, 1, batch*H*W, C] tensor.
    batch_size = unpack_mlir_attr(op.attributes["batch_size"])
    input_height = unpack_mlir_attr(op.attributes["input_height"])
    input_width = unpack_mlir_attr(op.attributes["input_width"])
    channels = unpack_mlir_attr(op.attributes["channels"])
    kernel_size = unpack_mlir_attr(op.attributes["kernel_size"])
    stride = unpack_mlir_attr(op.attributes["stride"])
    dilation = unpack_mlir_attr(op.attributes["dilation"])
    ceil_mode = unpack_mlir_attr(op.attributes["ceil_mode"])
    padding = unpack_mlir_attr(op.attributes["padding"])
    output_dtype = mlir_type_to_torch_dtype(op.results[0].type.element_type)

    # Unflatten to NHWC then transpose to NCHW for PyTorch pooling.
    nhwc = _ttnn_unflatten_nhwc(
        inputs["input"], batch_size, input_height, input_width, channels
    )
    nchw = nhwc.transpose(-2, -1).transpose(-3, -2)

    if isinstance(padding, (list, tuple)) and len(padding) == 4:
        top, left, bottom, right = padding
        if top == bottom and left == right:
            torch_padding = (top, left)
        else:
            nchw = torch.nn.functional.pad(
                nchw, [left, right, top, bottom], mode="constant", value=float("-inf")
            )
            torch_padding = 0
    else:
        torch_padding = padding

    result = torch.nn.MaxPool2d(
        kernel_size=kernel_size,
        stride=stride,
        padding=torch_padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )(nchw)

    # Transpose back to NHWC and flatten to [1, 1, batch*H_out*W_out, C].
    result = result.transpose(-3, -2).transpose(-2, -1).to(output_dtype)
    return _ttnn_flatten_nhwc(result)


def chisel_ttnn_avg_pool2d(op, inputs):
    # TTNN input/output is flat [1, 1, N*H*W, C]; unflatten to NHWC, run pool, re-flatten.
    batch_size = unpack_mlir_attr(op.attributes["batch_size"])
    input_height = unpack_mlir_attr(op.attributes["input_height"])
    input_width = unpack_mlir_attr(op.attributes["input_width"])
    channels = unpack_mlir_attr(op.attributes["channels"])
    nhwc = _ttnn_unflatten_nhwc(
        inputs["input"], batch_size, input_height, input_width, channels
    )
    result_nhwc = avg_pool2d_golden(
        input_tensor=nhwc,
        kernel=unpack_mlir_attr(op.attributes["kernel_size"]),
        stride=unpack_mlir_attr(op.attributes["stride"]),
        padding=unpack_mlir_attr(op.attributes["padding"]),
        ceil_mode=unpack_mlir_attr(op.attributes["ceil_mode"]),
        count_include_pad=_attr_get_value(
            op.attributes, "count_include_pad", default=True
        ),
    )
    return _ttnn_flatten_nhwc(result_nhwc)


def chisel_ttnn_max_pool2d_with_indices(op, inputs):
    # TTNN input/output is flat [1, 1, N*H*W, C]; unflatten to NHWC, run pool, re-flatten.
    batch_size = unpack_mlir_attr(op.attributes["batch_size"])
    input_height = unpack_mlir_attr(op.attributes["input_height"])
    input_width = unpack_mlir_attr(op.attributes["input_width"])
    channels = unpack_mlir_attr(op.attributes["channels"])
    nhwc = _ttnn_unflatten_nhwc(
        inputs["input"], batch_size, input_height, input_width, channels
    )
    # TTNN padding is [pad_H, pad_W]; expand to [top, left, bottom, right] for the golden.
    padding = unpack_mlir_attr(op.attributes["padding"])
    if len(padding) == 2:
        padding = [padding[0], padding[1], padding[0], padding[1]]
    values_nhwc, indices_nhwc = ttir_max_pool2d_with_indices(
        nhwc,
        op.attributes["kernel_size"],
        op.attributes["stride"],
        padding,
        op.attributes["dilation"],
        op.attributes["ceil_mode"],
        op.results[0].type.element_type,
    )
    indices_dtype = mlir_type_to_torch_dtype(op.results[1].type.element_type)
    return (
        _ttnn_flatten_nhwc(values_nhwc),
        _ttnn_flatten_nhwc(indices_nhwc).to(indices_dtype),
    )


def chisel_ttnn_conv2d(op, inputs):
    # TTNN input/output is flat [1, 1, N*H*W, C]; unflatten to NHWC, run conv, re-flatten.
    batch_size = unpack_mlir_attr(op.attributes["batch_size"])
    input_height = unpack_mlir_attr(op.attributes["input_height"])
    input_width = unpack_mlir_attr(op.attributes["input_width"])
    in_channels = unpack_mlir_attr(op.attributes["in_channels"])
    out_channels = unpack_mlir_attr(op.attributes["out_channels"])
    kh, kw = unpack_mlir_attr(op.attributes["kernel_size"])
    groups = unpack_mlir_attr(op.attributes["groups"])
    nhwc = _ttnn_unflatten_nhwc(
        inputs["input"], batch_size, input_height, input_width, in_channels
    )
    # TTNN weight is [1, 1, (C_in/groups)*kH*kW, C_out]; reshape to OIHW for torch conv2d.
    in_channels_per_group = in_channels // groups
    weight_oihw = (
        inputs["weight"]
        .reshape(in_channels_per_group, kh, kw, out_channels)
        .permute(3, 0, 1, 2)
    )
    result_nhwc = conv2d_golden(
        input_tensor=nhwc,
        weight=weight_oihw,
        bias=inputs["bias"],
        stride=op.attributes["stride"],
        padding=op.attributes["padding"],
        dilation=op.attributes["dilation"],
        groups=op.attributes["groups"],
        batch_dim=0,
        height_dim=1,
        width_dim=2,
        channel_dim=3,
    )
    return _ttnn_flatten_nhwc(result_nhwc)


def chisel_ttnn_conv3d(op, inputs):
    # TTNN conv3d receives a flat [1, 1, batch*D*H*W, C] tensor; unflatten to NDHWC.
    batch_size = unpack_mlir_attr(op.attributes["batch_size"])
    input_depth = unpack_mlir_attr(op.attributes["input_depth"])
    input_height = unpack_mlir_attr(op.attributes["input_height"])
    input_width = unpack_mlir_attr(op.attributes["input_width"])
    in_channels = unpack_mlir_attr(op.attributes["in_channels"])
    out_channels = unpack_mlir_attr(op.attributes["out_channels"])
    groups = unpack_mlir_attr(op.attributes["groups"])
    kernel_size = unpack_mlir_attr(op.attributes["kernel_size"])
    input_ndhwc = inputs["input"].reshape(
        batch_size, input_depth, input_height, input_width, in_channels
    )
    # TTNN weight is [K_D*K_H*K_W*(C_in/groups), C_out]; reshape to OIDHW for torch conv3d.
    kd, kh, kw = kernel_size
    in_channels_per_group = in_channels // groups
    weight_ncdhw = (
        inputs["weight"]
        .reshape(kd, kh, kw, in_channels_per_group, out_channels)
        .permute(4, 3, 0, 1, 2)
    )
    result_ndhwc = conv3d_golden(
        input_tensor=input_ndhwc,
        weight=weight_ncdhw,
        bias=inputs["bias"],
        stride=op.attributes["stride"],
        padding=op.attributes["padding"],
        groups=op.attributes["groups"],
        batch_dim=0,
        depth_dim=1,
        height_dim=2,
        width_dim=3,
        channel_dim=4,
        padding_mode=op.attributes["padding_mode"],
    )
    # TTNN result type is NDHWC (not flat), so return as-is.
    return result_ndhwc


def chisel_ttnn_conv_transpose2d(op, inputs):
    # TTNN input/output is flat [1, 1, N*H*W, C]; unflatten to NHWC, run conv, re-flatten.
    batch_size = unpack_mlir_attr(op.attributes["batch_size"])
    input_height = unpack_mlir_attr(op.attributes["input_height"])
    input_width = unpack_mlir_attr(op.attributes["input_width"])
    in_channels = unpack_mlir_attr(op.attributes["in_channels"])
    out_channels = unpack_mlir_attr(op.attributes["out_channels"])
    kh, kw = unpack_mlir_attr(op.attributes["kernel_size"])
    nhwc = inputs["input"].reshape(batch_size, input_height, input_width, in_channels)
    # TTNN weight is [1, 1, C_in*kH*kW, C_out]; reshape to IOHW for torch conv_transpose2d.
    weight_iohw = (
        inputs["weight"].reshape(in_channels, kh, kw, out_channels).permute(0, 3, 1, 2)
    )
    result_nhwc = conv_transpose2d_golden(
        input_tensor=nhwc,
        weight=weight_iohw,
        bias=inputs["bias"],
        stride=op.attributes["stride"],
        padding=op.attributes["padding"],
        output_padding=op.attributes["output_padding"],
        dilation=op.attributes["dilation"],
        groups=op.attributes["groups"],
        batch_dim=0,
        height_dim=1,
        width_dim=2,
        channel_dim=3,
    )
    n, h_out, w_out, c_out = result_nhwc.shape
    return result_nhwc.reshape(1, 1, n * h_out * w_out, c_out)


def chisel_ttnn_prepare_conv2d_weights(op, inputs):
    output_dtype = mlir_type_to_torch_dtype(op.results[0].type.element_type)
    out_shape = list(op.results[0].type.shape)
    # Permute [C_out, C_in, kH, kW] -> [C_in, kH, kW, C_out] then reshape to out_shape.
    perm = inputs["weight_tensor"].permute(1, 2, 3, 0)
    return perm.reshape(out_shape).to(output_dtype)


def chisel_ttnn_prepare_conv2d_bias(op, inputs):
    return (
        inputs["bias_tensor"]
        .clone()
        .to(mlir_type_to_torch_dtype(op.results[0].type.element_type))
    )


def chisel_ttnn_prepare_conv_transpose2d_weights(op, inputs):
    output_dtype = mlir_type_to_torch_dtype(op.results[0].type.element_type)
    out_shape = list(op.results[0].type.shape)
    # Permute [C_in, C_out, kH, kW] -> [C_in, kH, kW, C_out] then reshape to out_shape.
    perm = inputs["weight_tensor"].permute(0, 2, 3, 1)
    return perm.reshape(out_shape).to(output_dtype)


def chisel_ttnn_prepare_conv_transpose2d_bias(op, inputs):
    return (
        inputs["bias_tensor"]
        .clone()
        .to(mlir_type_to_torch_dtype(op.results[0].type.element_type))
    )


def chisel_ttnn_batch_norm_inference(op, inputs):
    return ttir_batch_norm_inference_golden(
        input_tensor=inputs["input"],
        scale=inputs["weight"],
        offset=inputs["bias"],
        mean=inputs["running_mean"],
        variance=inputs["running_var"],
        epsilon_attr=op.attributes["epsilon"],
        dimension_attr=1,
        output_type_mlir=op.results[0].type.element_type,
    )


def chisel_ttnn_batch_norm_training(op, inputs):
    output_type = op.results[0].type.element_type
    running_mean = inputs["running_mean"]
    running_var = inputs["running_var"]
    rm_shape = list(running_mean.shape)
    rv_shape = list(running_var.shape)
    result, updated_running_mean, updated_running_var = ttir_batch_norm_training_golden(
        input_tensor=inputs["input"],
        scale=inputs["weight"],
        offset=inputs["bias"],
        running_mean=torch.reshape(running_mean, [-1]),
        running_variance=torch.reshape(running_var, [-1]),
        epsilon_attr=op.attributes["epsilon"],
        dimension_attr=1,
        momentum_attr=op.attributes["momentum"],
        output_type_mlir=output_type,
        mean_output_type_mlir=output_type,
        variance_output_type_mlir=output_type,
    )
    return (
        result,
        torch.reshape(updated_running_mean, rm_shape),
        torch.reshape(updated_running_var, rv_shape),
    )


def chisel_ttnn_distributed_rms_norm(op, inputs):
    return ttir_distributed_rms_norm_golden(
        input=inputs["input"],
        weight=inputs["weight"],
        residual=inputs["residual"],
        cluster_axis_attr=op.attributes["cluster_axis"],
        epsilon_attr=op.attributes["epsilon"],
        output_type_mlir=op.results[0].type.element_type,
    )


def chisel_ttnn_scatter(op, inputs):
    return ttir_scatter_golden(
        input_tensor=inputs["input"],
        index=inputs["index"],
        source=inputs["source"],
        dim=op.attributes["dim"],
        scatter_reduce_type_attr=op.attributes["scatter_reduce_type"],
        output_type_mlir=op.results[0].type.element_type,
    )


def chisel_ttnn_scaled_dot_product_attention(op, inputs):
    return ttir_sdpa_golden(
        query=inputs["query"],
        key=inputs["key"],
        value=inputs["value"],
        attention_mask=inputs["attention_mask"],
        is_causal_attr=_attr_get(op.attributes, "is_causal"),
        scale_attr=_attr_get(op.attributes, "scale"),
        output_type_mlir=op.results[0].type.element_type,
    )


def chisel_ttnn_scaled_dot_product_attention_decode(op, inputs):
    return sdpa_decode_golden(
        query=inputs["query"],
        key=inputs["key"],
        value=inputs["value"],
        cur_pos_tensor=inputs["cur_pos_tensor"],
        attention_mask=inputs["attention_mask"],
        is_causal=_attr_get_value(op.attributes, "is_causal", default=True),
        scale=_attr_get_value(op.attributes, "scale"),
    )


def chisel_ttnn_paged_scaled_dot_product_attention_decode(op, inputs):
    return ttir_paged_sdpa_decode_golden(
        query=inputs["query"],
        key=inputs["key"],
        value=inputs["value"],
        page_table=inputs["page_table"],
        output=None,
        is_causal_attr=_attr_get(op.attributes, "is_causal"),
        attention_mask=inputs["attention_mask"],
        cur_pos_tensor=inputs["cur_pos_tensor"],
        attention_sink=inputs["attention_sink"],
        scale_attr=_attr_get(op.attributes, "scale"),
        output_type_mlir=op.results[0].type.element_type,
    )


def chisel_ttnn_fill_cache(op, inputs):
    return fill_cache_golden(cache_tensor=inputs["cache"], input_tensor=inputs["input"])


def chisel_ttnn_update_cache(op, inputs):
    return update_cache_golden(
        cache_tensor=inputs["cache"],
        update_tensor=inputs["input"],
        indices_tensor=inputs["update_index"],
        batch_offset=op.attributes["batch_offset"],
    )


def chisel_ttnn_paged_fill_cache(op, inputs):
    cache = inputs["cache"]
    if cache.device.type == "meta":
        return cache.clone()
    return ttir_paged_fill_cache_golden(
        cache_tensor=cache,
        input_tensor=inputs["input"],
        page_table_tensor=inputs["page_table"],
        batch_idx_tensor=inputs["batch_idx_tensor"],
        output_type_mlir=op.operands[0].type.element_type,
    )


def chisel_ttnn_paged_update_cache(op, inputs):
    cache = inputs["cache"]
    if cache.device.type == "meta":
        return cache.clone()
    return ttir_paged_update_cache_golden(
        cache_tensor=cache,
        input_tensor=inputs["input"],
        update_index_tensor=inputs["update_index"],
        share_cache_attr=_attr_get(op.attributes, "share_cache", default=False),
        page_table_tensor=inputs["page_table"],
        output_type_mlir=op.operands[0].type.element_type,
    )


def chisel_ttnn_all_reduce(op, inputs):
    return ttir_all_reduce_golden(
        input=inputs["input"],
        reduce_type_attr=op.attributes["reduce_type"],
        cluster_axis_attr=op.attributes["cluster_axis"],
        output_type_mlir=op.results[0].type.element_type,
    )


def chisel_ttnn_all_to_all_dispatch(op, inputs):
    return ttir_all_to_all_dispatch_golden(
        input_tensor=inputs["input_tensor"],
        expert_indices=inputs["expert_indices"],
        expert_mapping=inputs["expert_mapping"],
        num_devices_attr=op.attributes["num_devices"],
        cluster_axis_attr=op.attributes["cluster_axis"],
        dispatched_type_mlir=op.results[0].type.element_type,
        metadata_type_mlir=op.results[1].type.element_type,
    )


def chisel_ttnn_all_to_all_combine(op, inputs):
    return all_to_all_combine_golden(
        input_tensor=inputs["input_tensor"],
        expert_metadata=inputs["expert_metadata"],
        expert_mapping=inputs["expert_mapping"],
        num_devices=unpack_mlir_attr(op.attributes["num_devices"]),
        cluster_axis=unpack_mlir_attr(op.attributes["cluster_axis"]),
        num_experts_per_tok=unpack_mlir_attr(op.attributes["num_experts_per_tok"]),
    )


def chisel_ttnn_concatenate_heads(op, inputs):
    return ttir_concatenate_heads_golden(
        input_tensor=inputs["input"], output_type_mlir=op.results[0].type.element_type
    )


def chisel_ttnn_split_query_key_value_and_split_heads(op, inputs):
    return ttir_split_query_key_value_and_split_heads_golden(
        input_tensor=inputs["input_tensor"],
        kv_input_tensor=inputs["kv_input_tensor"],
        num_heads_attr=op.attributes["num_heads"],
        num_kv_heads_attr=_attr_get(op.attributes, "num_kv_heads"),
        transpose_key_attr=op.attributes["transpose_key"],
        query_output_type_mlir=op.results[0].type.element_type,
        key_output_type_mlir=op.results[1].type.element_type,
        value_output_type_mlir=op.results[2].type.element_type,
    )


def chisel_ttnn_topk_router_gpt(op, inputs):
    return ttir_topk_router_gpt_golden(
        input_tensor=inputs["input"],
        weight_tensor=inputs["weight"],
        bias_tensor=inputs["bias"],
        k_attr=op.attributes["k"],
        _num_experts_attr=op.attributes["num_experts"],
        output_type_mlir=op.results[0].type.element_type,
    )


def chisel_ttnn_moe_expert_token_remap(op, inputs):
    return ttir_moe_expert_token_remap_golden(
        topk_tensor=inputs["topk_tensor"],
        expert_mapping=inputs["expert_mapping"],
        expert_metadata=inputs["expert_metadata"],
        reduction_size=unpack_mlir_attr(op.attributes["reduction_size"]),
    )


def chisel_ttnn_sparse_matmul(op, inputs):
    return sparse_matmul_golden(
        a=inputs["a"],
        b=inputs["b"],
        sparsity=inputs["sparsity"],
        is_input_a_sparse=unpack_mlir_attr(op.attributes["is_input_a_sparse"]),
        is_input_b_sparse=unpack_mlir_attr(op.attributes["is_input_b_sparse"]),
        nnz=_attr_get_value(op.attributes, "nnz"),
    )


def chisel_ttnn_upsample(op, inputs):
    input_tensor = inputs["input"]
    scale_factor = unpack_mlir_attr(op.attributes["scale_factor"])
    mode = unpack_mlir_attr(op.attributes["mode"])
    # NHWC -> NCHW for torch.interpolate.
    nchw = input_tensor.permute(0, 3, 1, 2).float()
    h_in, w_in = nchw.shape[2], nchw.shape[3]
    if isinstance(scale_factor, (list, tuple)):
        scale_h, scale_w = scale_factor[0], scale_factor[1]
    else:
        scale_h = scale_w = scale_factor
    output_size = (int(h_in * scale_h), int(w_in * scale_w))
    # align_corners=False matches the default TTNN upsample behaviour.
    align_corners = False if mode in ("bilinear", "bicubic", "linear") else None
    interp_kwargs = (
        {"align_corners": align_corners} if align_corners is not None else {}
    )
    result_nchw = torch.nn.functional.interpolate(
        nchw, size=output_size, mode=mode, **interp_kwargs
    )
    return result_nchw.permute(0, 2, 3, 1).to(input_tensor.dtype)


def chisel_ttnn_pow_scalar(op, inputs):
    exponent = unpack_mlir_attr(op.attributes["rhs"])
    output_dtype = mlir_type_to_torch_dtype(op.results[0].type.element_type)
    return torch.pow(inputs["lhs"], exponent).to(output_dtype)


def _chisel_mesh_shape_from_op(op) -> list:
    """Return [rows, cols] mesh shape from the device operand, or [1, 1] if absent."""
    for operand in op.operands:
        try:
            defining_op = operand.owner
            if "mesh_shape" in defining_op.attributes:
                ms = ttnn.ir.MeshShapeAttr.maybe_downcast(
                    defining_op.attributes["mesh_shape"]
                )
                return [ms.y, ms.x]
        except Exception:
            continue
    return [1, 1]


def chisel_ttnn_arange(op, inputs):
    # TTNN ArangeOp always produces a 1D result - no arange_dimension concept.
    start = unpack_mlir_attr(op.attributes["start"])
    end = unpack_mlir_attr(op.attributes["end"])
    step = unpack_mlir_attr(op.attributes["step"])
    output_shape = list(op.results[0].type.shape)
    output_dtype = mlir_type_to_torch_dtype(op.results[0].type.element_type)
    mesh_shape = _chisel_mesh_shape_from_op(op)
    result = torch.arange(start=start, end=end, step=step).to(output_dtype)
    if list(result.shape) != output_shape:
        result = result.reshape(output_shape)
    num_devices = mesh_shape[0] * mesh_shape[1]
    return GoldenMapTensor({i: result.clone() for i in range(num_devices)}, mesh_shape)


def chisel_ttnn_full(op, inputs):
    shape = ttnn.ir.ShapeAttr.maybe_downcast(op.attributes["shape"]).shape
    return ttir_full_golden(
        shape_attr=shape,
        fill_value_attr=op.attributes["fill_value"],
        mesh_shape_attr=_chisel_mesh_shape_from_op(op),
        output_type_mlir=op.results[0].type.element_type,
    )


def chisel_ttnn_ones(op, inputs):
    shape = ttnn.ir.ShapeAttr.maybe_downcast(op.attributes["shape"]).shape
    return ttir_ones_golden(
        shape=shape,
        mesh_shape_attr=_chisel_mesh_shape_from_op(op),
        output_type_mlir=op.results[0].type.element_type,
    )


def chisel_ttnn_zeros(op, inputs):
    shape = ttnn.ir.ShapeAttr.maybe_downcast(op.attributes["shape"]).shape
    return ttir_zeros_golden(
        shape=shape,
        mesh_shape_attr=_chisel_mesh_shape_from_op(op),
        output_type_mlir=op.results[0].type.element_type,
    )


def chisel_ttnn_empty(op, inputs):
    shape = ttnn.ir.ShapeAttr.maybe_downcast(op.attributes["shape"]).shape
    return ttir_zeros_golden(
        shape=shape,
        mesh_shape_attr=_chisel_mesh_shape_from_op(op),
        output_type_mlir=op.results[0].type.element_type,
    )


def chisel_ttnn_dump_tensor(op, inputs):
    return ()


def chisel_ttnn_rand(op, inputs):
    size = ttnn.ir.ShapeAttr.maybe_downcast(op.attributes["size"]).shape
    return ttir_rand_golden(
        size=size,
        low=op.attributes["low"],
        high=op.attributes["high"],
        seed=op.attributes["seed"],
        mesh_shape_attr=_chisel_mesh_shape_from_op(op),
        output_type_mlir=op.results[0].type.element_type,
    )


CHISEL_GOLDEN_MAPPINGS: Dict[type, Callable] = {
    # Unary ops
    ttnn.AbsOp: _chisel_unary(ttnn_abs_golden),
    ttnn.CbrtOp: _chisel_unary(ttnn_cbrt_golden),
    ttnn.CeilOp: _chisel_unary(ttnn_ceil_golden),
    ttnn.CosOp: _chisel_unary(ttnn_cos_golden),
    ttnn.AcosOp: _chisel_unary(ttnn_acos_golden),
    ttnn.ErfOp: _chisel_unary(ttnn_erf_golden),
    ttnn.ErfcOp: _chisel_unary(ttnn_erfc_golden),
    ttnn.ExpOp: _chisel_unary(ttnn_exp_golden),
    ttnn.FloorOp: _chisel_unary(ttnn_floor_golden),
    ttnn.GeluOp: _chisel_unary(ttnn_gelu_golden),
    ttnn.IsFiniteOp: _chisel_unary(ttnn_isfinite_golden),
    ttnn.NegOp: _chisel_unary(ttnn_neg_golden),
    ttnn.TanOp: _chisel_unary(ttnn_tan_golden),
    ttnn.AtanOp: _chisel_unary(ttnn_atan_golden),
    ttnn.TanhOp: _chisel_unary(ttnn_tanh_golden),
    ttnn.ReciprocalOp: _chisel_unary(ttnn_reciprocal_golden),
    ttnn.ReluOp: _chisel_unary(ttnn_relu_golden),
    ttnn.Relu6Op: _chisel_unary(ttnn_relu6_golden),
    ttnn.RsqrtOp: _chisel_unary(ttnn_rsqrt_golden),
    ttnn.SigmoidOp: _chisel_unary(ttnn_sigmoid_golden),
    ttnn.SignOp: _chisel_unary(ttnn_sign_golden),
    ttnn.SiluOp: _chisel_unary(ttnn_silu_golden),
    ttnn.SinOp: _chisel_unary(ttnn_sin_golden),
    ttnn.AsinOp: _chisel_unary(ttnn_asin_golden),
    ttnn.SqrtOp: _chisel_unary(ttnn_sqrt_golden),
    ttnn.LogOp: _chisel_unary(ttnn_log_golden),
    ttnn.Log1pOp: _chisel_unary(ttnn_log1p_golden),
    ttnn.Expm1Op: _chisel_unary(ttnn_expm1_golden),
    ttnn.MishOp: _chisel_unary(ttnn_mish_golden),
    ttnn.LogicalNotOp: _chisel_unary(ttnn_logical_not_golden),
    ttnn.BitwiseNotOp: _chisel_unary(ttnn_bitwise_not_golden),
    ttnn.ToDeviceOp: _chisel_unary(ttnn_to_device_golden),
    ttnn.FromDeviceOp: _chisel_unary(ttnn_from_device_golden),
    # Binary ops
    ttnn.AddOp: _chisel_binary(ttnn_add_golden),
    ttnn.Atan2Op: _chisel_binary(ttnn_atan2_golden),
    ttnn.MultiplyOp: _chisel_binary(ttnn_multiply_golden),
    ttnn.SubtractOp: _chisel_binary(ttnn_subtract_golden),
    ttnn.DivideOp: _chisel_binary(ttnn_divide_golden),
    ttnn.MaximumOp: _chisel_binary(ttnn_maximum_golden),
    ttnn.MinimumOp: _chisel_binary(ttnn_minimum_golden),
    ttnn.RemainderOp: _chisel_binary(ttnn_remainder_golden),
    ttnn.PowTensorOp: _chisel_binary(ttnn_pow_tensor_golden),
    ttnn.EqualOp: _chisel_binary(ttnn_eq_golden),
    ttnn.NotEqualOp: _chisel_binary(ttnn_ne_golden),
    ttnn.GreaterEqualOp: _chisel_binary(ttnn_ge_golden),
    ttnn.GreaterThanOp: _chisel_binary(ttnn_gt_golden),
    ttnn.LessEqualOp: _chisel_binary(ttnn_le_golden),
    ttnn.LessThanOp: _chisel_binary(ttnn_lt_golden),
    ttnn.LogicalAndOp: _chisel_binary(ttnn_logical_and_golden),
    ttnn.LogicalOrOp: _chisel_binary(ttnn_logical_or_golden),
    ttnn.LogicalXorOp: _chisel_binary(ttnn_logical_xor_golden),
    ttnn.LogicalLeftShiftOp: _chisel_binary(
        ttnn_logical_left_shift_golden, rhs_kwarg="shift_tensor"
    ),
    ttnn.LogicalRightShiftOp: _chisel_binary(
        ttnn_logical_right_shift_golden, rhs_kwarg="shift_tensor"
    ),
    ttnn.BitwiseAndOp: _chisel_binary(ttnn_bitwise_and_golden),
    ttnn.BitwiseOrOp: _chisel_binary(ttnn_bitwise_or_golden),
    ttnn.BitwiseXorOp: _chisel_binary(ttnn_bitwise_xor_golden),
    # Ternary ops
    ttnn.WhereOp: chisel_ttnn_where,
    ttnn.ClampTensorOp: chisel_ttnn_clamp_tensor,
    # Type / layout ops
    ttnn.TypecastOp: chisel_ttnn_typecast,
    ttnn.ToLayoutOp: chisel_ttnn_to_layout,
    # Unary + scalar attrs
    ttnn.LeakyReluOp: chisel_ttnn_leaky_relu,
    ttnn.ClampScalarOp: chisel_ttnn_clamp_scalar,
    ttnn.RepeatInterleaveOp: chisel_ttnn_repeat_interleave,
    # Binary + attrs
    ttnn.MatmulOp: chisel_ttnn_matmul,
    # Variadic + attrs
    ttnn.ConcatOp: chisel_ttnn_concat,
    ttnn.RepeatOp: chisel_ttnn_repeat,
    # Optional tensors + attrs
    ttnn.LinearOp: chisel_ttnn_linear,
    ttnn.LayerNormOp: chisel_ttnn_layer_norm,
    ttnn.LayerNormPreAllGatherOp: chisel_ttnn_layer_norm_pre_all_gather,
    ttnn.LayerNormPostAllGatherOp: chisel_ttnn_layer_norm_post_all_gather,
    ttnn.RMSNormOp: chisel_ttnn_rms_norm,
    # CCL ops
    ttnn.DistributeTensorOp: chisel_ttnn_distribute_tensor,
    ttnn.AggregateTensorOp: chisel_ttnn_aggregate_tensor,
    ttnn.AllGatherOp: chisel_ttnn_all_gather,
    ttnn.ReduceScatterOp: chisel_ttnn_reduce_scatter,
    ttnn.AllReduceAsyncOp: chisel_ttnn_all_reduce_async,
    # Index-based
    ttnn.GatherOp: chisel_ttnn_gather,
    # Attention
    ttnn.PagedFlashMultiLatentAttentionDecodeOp: chisel_ttnn_paged_flash_multi_latent_attention_decode,
    # Layout / memory ops
    ttnn.AssignOp: chisel_ttnn_assign,
    ttnn.ToMemoryConfigOp: chisel_ttnn_to_memory_config,
    ttnn.DeallocateOp: chisel_ttnn_deallocate,
    # Reduction ops
    ttnn.SumOp: chisel_ttnn_sum,
    ttnn.MeanOp: chisel_ttnn_mean,
    ttnn.MaxOp: chisel_ttnn_max,
    ttnn.MinOp: chisel_ttnn_min,
    ttnn.ProdOp: chisel_ttnn_prod,
    ttnn.ArgMaxOp: chisel_ttnn_argmax,
    ttnn.CumSumOp: chisel_ttnn_cumsum,
    ttnn.SortOp: chisel_ttnn_sort,
    # Shape/layout ops
    ttnn.TransposeOp: chisel_ttnn_transpose,
    ttnn.ReshapeOp: chisel_ttnn_reshape,
    ttnn.PermuteOp: chisel_ttnn_permute,
    ttnn.SliceStaticOp: chisel_ttnn_slice_static,
    ttnn.SliceDynamicOp: chisel_ttnn_slice_dynamic,
    ttnn.TopKOp: chisel_ttnn_topk,
    ttnn.PadOp: chisel_ttnn_pad,
    ttnn.SoftmaxOp: chisel_ttnn_softmax,
    # NN / activation ops
    ttnn.HardsigmoidOp: chisel_ttnn_hardsigmoid,
    ttnn.EmbeddingOp: chisel_ttnn_embedding,
    ttnn.EmbeddingBackwardOp: chisel_ttnn_embedding_backward,
    ttnn.GeluBackwardOp: chisel_ttnn_gelu_backward,
    ttnn.DropoutOp: chisel_ttnn_dropout,
    # Conv/Pool ops
    ttnn.GlobalAvgPool2dOp: chisel_ttnn_global_avg_pool2d,
    ttnn.MaxPool2dOp: chisel_ttnn_max_pool2d,
    ttnn.AvgPool2dOp: chisel_ttnn_avg_pool2d,
    ttnn.MaxPool2dWithIndicesOp: chisel_ttnn_max_pool2d_with_indices,
    ttnn.Conv2dOp: chisel_ttnn_conv2d,
    ttnn.Conv3dOp: chisel_ttnn_conv3d,
    ttnn.ConvTranspose2dOp: chisel_ttnn_conv_transpose2d,
    ttnn.PrepareConv2dWeightsOp: chisel_ttnn_prepare_conv2d_weights,
    ttnn.PrepareConv2dBiasOp: chisel_ttnn_prepare_conv2d_bias,
    ttnn.PrepareConvTranspose2dWeightsOp: chisel_ttnn_prepare_conv_transpose2d_weights,
    ttnn.PrepareConvTranspose2dBiasOp: chisel_ttnn_prepare_conv_transpose2d_bias,
    # BatchNorm / DistRMSNorm / Scatter
    ttnn.BatchNormInferenceOp: chisel_ttnn_batch_norm_inference,
    ttnn.BatchNormTrainingOp: chisel_ttnn_batch_norm_training,
    ttnn.DistributedRMSNormOp: chisel_ttnn_distributed_rms_norm,
    ttnn.ScatterOp: chisel_ttnn_scatter,
    # SDPA / Attention ops
    ttnn.ScaledDotProductAttentionOp: chisel_ttnn_scaled_dot_product_attention,
    ttnn.ScaledDotProductAttentionDecodeOp: chisel_ttnn_scaled_dot_product_attention_decode,
    ttnn.PagedScaledDotProductAttentionDecodeOp: chisel_ttnn_paged_scaled_dot_product_attention_decode,
    # Cache ops
    ttnn.FillCacheOp: chisel_ttnn_fill_cache,
    ttnn.UpdateCacheOp: chisel_ttnn_update_cache,
    ttnn.PagedFillCacheOp: chisel_ttnn_paged_fill_cache,
    ttnn.PagedUpdateCacheOp: chisel_ttnn_paged_update_cache,
    # CCL ops
    ttnn.AllReduceOp: chisel_ttnn_all_reduce,
    ttnn.AllToAllDispatchOp: chisel_ttnn_all_to_all_dispatch,
    ttnn.AllToAllCombineOp: chisel_ttnn_all_to_all_combine,
    # NLP / attention-specific ops
    ttnn.ConcatenateHeadsOp: chisel_ttnn_concatenate_heads,
    ttnn.SplitQueryKeyValueAndSplitHeadsOp: chisel_ttnn_split_query_key_value_and_split_heads,
    ttnn.TopKRouterGptOp: chisel_ttnn_topk_router_gpt,
    ttnn.MoeExpertTokenRemapOp: chisel_ttnn_moe_expert_token_remap,
    ttnn.SparseMatmulOp: chisel_ttnn_sparse_matmul,
    # Remaining ops
    ttnn.UpsampleOp: chisel_ttnn_upsample,
    ttnn.PowScalarOp: chisel_ttnn_pow_scalar,
    # Tensor creation stubs
    ttnn.ArangeOp: chisel_ttnn_arange,
    ttnn.FullOp: chisel_ttnn_full,
    ttnn.OnesOp: chisel_ttnn_ones,
    ttnn.ZerosOp: chisel_ttnn_zeros,
    ttnn.RandOp: chisel_ttnn_rand,
    # Tensor creation ops
    ttnn.EmptyOp: chisel_ttnn_empty,
    # Side-effect-only ops (no result)
    ttnn.DumpTensorOp: chisel_ttnn_dump_tensor,
    # Debug ops
    debug.AnnotateOp: chisel_debug_annotate,
    debug.RegionStartOp: chisel_debug_region_start,
    debug.RegionEndOp: chisel_debug_region_end,
}


def get_chisel_golden_function(op_class: type) -> Optional[Callable]:
    """Return the chisel golden for `op_class`, or None if unregistered."""
    return CHISEL_GOLDEN_MAPPINGS.get(op_class, None)
