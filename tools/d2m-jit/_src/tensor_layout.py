# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""The `Layout` descriptor and the dtype / mem_space vocabulary.

The MLIR bindings are imported **lazily** (`_mlir`), not at module scope, so
this module imports in an environment with no tt-metal build. That is what lets
the pure-torch simulator in `_src/sim/` -- which needs only the descriptor
fields, never the `build_*` methods -- run with no compiler and no runtime
extension. When the bindings are present nothing changes: `dtype` / `mem_space`
resolve to the same `ttcore` enum members as before, so the device path is
unaffected.

When the bindings are absent, `dtype` / `mem_space` resolve to the pure-Python
`_DataType` / `_MemorySpace` mirrors below, which carry the same member names,
integer values, and `str()` spellings. Only the simulator ever sees those, and
it keys off `.name` (see `_src/sim/tensors.py`).
"""

import enum
from types import SimpleNamespace


# --- lazy MLIR bindings ------------------------------------------------------

# None = not yet probed; False = probed and unavailable.
_MLIR_CACHE = None


def _try_mlir():
    """The MLIR handles this module builds types with, or None if unavailable."""
    global _MLIR_CACHE
    if _MLIR_CACHE is None:
        try:
            from ttmlir.ir import (
                BF16Type,
                DenseIntElementsAttr,
                F16Type,
                F32Type,
                IntegerType,
                RankedTensorType,
            )
            from ttmlir.dialects import ttcore, d2m
        except ImportError:
            _MLIR_CACHE = False
        else:
            _MLIR_CACHE = SimpleNamespace(
                BF16Type=BF16Type,
                DenseIntElementsAttr=DenseIntElementsAttr,
                F16Type=F16Type,
                F32Type=F32Type,
                IntegerType=IntegerType,
                RankedTensorType=RankedTensorType,
                ttcore=ttcore,
                d2m=d2m,
            )
    return None if _MLIR_CACHE is False else _MLIR_CACHE


def _mlir():
    """As `_try_mlir`, but raises if the bindings are unavailable."""
    mlir = _try_mlir()
    if mlir is None:
        raise ImportError(
            "building MLIR types from a Layout requires the ttmlir bindings, "
            "which are not importable in this environment; only the pure-Python "
            "simulator (`import d2m_jit.sim`) runs without them"
        )
    return mlir


def _ttcore_or_none():
    mlir = _try_mlir()
    return None if mlir is None else mlir.ttcore


# --- dtype / mem_space vocabulary --------------------------------------------


class _DataType(enum.IntEnum):
    """Pure-Python mirror of `ttcore.DataType` (names/values/`str` match).

    Used only when the MLIR bindings are unavailable; see the module docstring.
    """

    Float32 = 0
    Float16 = 1
    BFloat16 = 2
    UInt32 = 9
    Int32 = 12

    def __str__(self):
        return {
            "Float32": "f32",
            "Float16": "f16",
            "BFloat16": "bf16",
            "UInt32": "u32",
            "Int32": "si32",
        }[self.name]


class _MemorySpace(enum.IntEnum):
    """Pure-Python mirror of `ttcore.MemorySpace` (names/values/`str` match)."""

    System = 0
    SystemMMIO = 1
    DeviceDRAM = 2
    DeviceL1 = 3

    def __str__(self):
        return {
            "System": "system",
            "SystemMMIO": "mmio",
            "DeviceDRAM": "dram",
            "DeviceL1": "l1",
        }[self.name]


def _data_type_name(dtype):
    """Canonical `ttcore.DataType` member name for `dtype`, or None."""
    if getattr(dtype, "name", None) in _DataType.__members__:
        return dtype.name
    s = str(dtype)
    if s in {"torch.float32", "fp32"}:
        return "Float32"
    if s in {"torch.float16", "fp16"}:
        return "Float16"
    if s in {"torch.bfloat16", "bf16"}:
        return "BFloat16"
    if s in {"torch.uint32", "uint32", "u32"}:
        return "UInt32"
    if s in {"torch.int32", "int32", "i32", "si32"}:
        return "Int32"
    return None


def _to_data_type(dtype):
    ttcore = _ttcore_or_none()
    if ttcore is not None and isinstance(dtype, ttcore.DataType):
        return dtype
    name = _data_type_name(dtype)
    if name is None:
        raise TypeError(f"Unsupported dtype {dtype}")
    return _DataType[name] if ttcore is None else getattr(ttcore.DataType, name)


def _mem_space_name(mem_space):
    """Canonical `ttcore.MemorySpace` member name for `mem_space`, or None."""
    if getattr(mem_space, "name", None) in _MemorySpace.__members__:
        return mem_space.name
    if mem_space in {"l1", "sram"}:
        return "DeviceL1"
    if mem_space == "dram":
        return "DeviceDRAM"
    return None


def _to_mem_space(mem_space):
    ttcore = _ttcore_or_none()
    if ttcore is not None and isinstance(mem_space, ttcore.MemorySpace):
        return mem_space
    name = _mem_space_name(mem_space)
    if name is None:
        raise TypeError(f"Unsupported mem_space {mem_space}")
    return _MemorySpace[name] if ttcore is None else getattr(ttcore.MemorySpace, name)


# Public dtype constants. Pass to `dtype=` on Layout / tilize / untilize
# instead of strings ("fp32", "bf16", ...). The strings are still accepted.
float32 = _to_data_type("fp32")
float16 = _to_data_type("fp16")
bfloat16 = _to_data_type("bf16")
uint32 = _to_data_type("u32")
int32 = _to_data_type("i32")


def _derive_blocked_grid_shape(logical_shape, block_shape, tiled):
    assert len(logical_shape) == len(block_shape)
    s = list(logical_shape)
    if tiled:
        for i in range(len(s)):
            s[i] = (s[i] + 31) // 32

    out = []
    for ls, bs in zip(s, block_shape):
        assert ls % bs == 0
        out.append(ls // bs)
    return out


class Layout:
    """Pure layout descriptor: shape + dtype + block/grid/tiling/mem_space.

    Has no association with any host buffer. Builds the various MLIR
    types/values needed to embed this layout in a host or device tensor.
    """

    def __init__(
        self,
        shape,
        dtype,
        block_shape,
        grid_shape=None,
        tiled=True,
        collapse=True,
        mem_space="l1",
    ):
        self.logical_shape = list(shape)
        self.dtype = _to_data_type(dtype)
        self.block_shape = list(block_shape)
        self.blocked_grid_shape = _derive_blocked_grid_shape(
            self.logical_shape, self.block_shape, tiled
        )
        self.grid_shape = (
            list(self.blocked_grid_shape) if grid_shape is None else list(grid_shape)
        )
        self.tiled = tiled
        self.collapse = collapse
        self.mem_space = _to_mem_space(mem_space)
        self._cached_layout = None

    def replace(self, **overrides) -> "Layout":
        """Return a new Layout copying self's fields, overriding any
        keyword in `overrides`."""
        fields = dict(
            shape=self.logical_shape,
            dtype=self.dtype,
            block_shape=self.block_shape,
            grid_shape=self.grid_shape,
            tiled=self.tiled,
            collapse=self.collapse,
            mem_space=self.mem_space,
        )
        fields.update(overrides)
        return Layout(**fields)

    def get_tile_shape(self):
        return [32, 32] if self.tiled else []

    def get_scalar_type(self, ctx):
        mlir = _mlir()
        if self.dtype.name == "Float32":
            return mlir.F32Type.get(ctx)
        if self.dtype.name == "Float16":
            return mlir.F16Type.get(ctx)
        if self.dtype.name == "BFloat16":
            return mlir.BF16Type.get(ctx)
        if self.dtype.name == "UInt32":
            return mlir.IntegerType.get_unsigned(32, ctx)
        if self.dtype.name == "Int32":
            return mlir.IntegerType.get_signed(32, ctx)
        raise TypeError(f"Unsupported data type {self.dtype}")

    def get_host_elem_type(self, ctx):
        return self.get_scalar_type(ctx)

    def get_device_elem_type(self, ctx):
        elem_type = self.get_scalar_type(ctx)
        if self.tiled:
            tile_shape = self.get_tile_shape()
            elem_type = _mlir().ttcore.ir.TileType.get(
                ctx, tile_shape[0], tile_shape[1], self.dtype
            )
        return elem_type

    def get_device_shape(self, ctx, grid_shape):
        layout = self.build_metal_layout(ctx)
        metal_layout = _mlir().ttcore.ir.MetalLayoutAttr.maybe_downcast(layout)
        return metal_layout.getDeviceShape(grid_shape, self.get_tile_shape())

    def build_host_tensor_type(self, ctx):
        return _mlir().RankedTensorType.get(
            self.logical_shape, self.get_host_elem_type(ctx)
        )

    def build_metal_layout(self, ctx):
        if self._cached_layout is not None:
            return self._cached_layout

        mlir = _mlir()
        ttcore = mlir.ttcore
        if self.collapse:
            self._cached_layout = ttcore.ir.MetalLayoutAttr.get(
                ctx,
                list(self.logical_shape),
                int(self.mem_space),
                int(ttcore.TensorMemoryLayout.Sharded),
            )
        else:
            empty_interval_type = mlir.RankedTensorType.get(
                [0, 2], mlir.IntegerType.get_signless(64)
            )
            empty_collapse_intervals = mlir.DenseIntElementsAttr.get(
                empty_interval_type, []
            )
            self._cached_layout = ttcore.ir.MetalLayoutAttr.get(
                ctx,
                list(self.logical_shape),
                int(self.mem_space),
                int(ttcore.TensorMemoryLayout.Sharded),
                empty_collapse_intervals,
                [],
            )

        return self._cached_layout

    def build_device_tensor_type(self, ctx, blocked=False):
        grid_shape = self.blocked_grid_shape if blocked else self.grid_shape
        layout = self.build_metal_layout(ctx)
        elem_type = self.get_device_elem_type(ctx)
        device_shape = self.get_device_shape(ctx, grid_shape)
        return _mlir().RankedTensorType.get(device_shape, elem_type, encoding=layout)

    def build_to_device(self, ctx, val):
        d2m = _mlir().d2m
        output_type = self.build_device_tensor_type(ctx)
        output = d2m.empty(output_type)
        res = d2m.ToLayoutOp([output_type], val, output).result
        return self.build_blocked_view(ctx, res)

    def build_blocked_view(self, ctx, val):
        if self.blocked_grid_shape == self.grid_shape:
            return val
        d2m = _mlir().d2m
        device_shape = self.get_device_shape(ctx, self.grid_shape)
        blocked_device_shape = self.get_device_shape(ctx, self.blocked_grid_shape)
        blocked_type = self.build_device_tensor_type(ctx, blocked=True)
        reblock_map = d2m.ir.calculate_reblock_map(
            device_shape, blocked_device_shape, ctx
        )
        return d2m.ViewLayoutOp(blocked_type, val, reblock_map).result

    def build_device_view(self, ctx, val):
        if self.blocked_grid_shape == self.grid_shape:
            return val
        d2m = _mlir().d2m
        device_shape = self.get_device_shape(ctx, self.grid_shape)
        blocked_device_shape = self.get_device_shape(ctx, self.blocked_grid_shape)
        device_type = self.build_device_tensor_type(ctx, blocked=False)
        reblock_map = d2m.ir.calculate_reblock_map(
            blocked_device_shape, device_shape, ctx
        )
        return d2m.ViewLayoutOp(device_type, val, reblock_map).result

    def build_from_device(self, ctx, val):
        d2m = _mlir().d2m
        output_type = self.build_host_tensor_type(ctx)
        output = d2m.empty(output_type)
        return d2m.ToLayoutOp([output_type], val, output).result
