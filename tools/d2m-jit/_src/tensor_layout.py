# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from ttmlir.ir import *
from ttmlir.dialects import ttcore, d2m


# Public dtype constants. Pass to `dtype=` on Layout / tilize / untilize
# instead of strings ("fp32", "bf16", ...). The strings are still accepted.
float32 = ttcore.DataType.Float32
float16 = ttcore.DataType.Float16
bfloat16 = ttcore.DataType.BFloat16


def _to_data_type(dtype):
    if isinstance(dtype, ttcore.DataType):
        return dtype
    s = str(dtype)
    if s in {"torch.float32", "fp32"}:
        return ttcore.DataType.Float32
    if s in {"torch.float16", "fp16"}:
        return ttcore.DataType.Float16
    if s in {"torch.bfloat16", "bf16"}:
        return ttcore.DataType.BFloat16
    raise TypeError(f"Unsupported dtype {dtype}")


def _to_mem_space(mem_space):
    if isinstance(mem_space, ttcore.MemorySpace):
        return mem_space
    if mem_space in {"l1", "sram"}:
        return ttcore.MemorySpace.DeviceL1
    if mem_space == "dram":
        return ttcore.MemorySpace.DeviceDRAM
    raise TypeError(f"Unsupported mem_space {mem_space}")


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
        mem_space=ttcore.MemorySpace.DeviceL1,
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
        if self.dtype == ttcore.DataType.Float32:
            return F32Type.get(ctx)
        if self.dtype == ttcore.DataType.Float16:
            return F16Type.get(ctx)
        if self.dtype == ttcore.DataType.BFloat16:
            return BF16Type.get(ctx)
        raise TypeError(f"Unsupported data type {self.dtype}")

    def get_host_elem_type(self, ctx):
        return self.get_scalar_type(ctx)

    def get_device_elem_type(self, ctx):
        elem_type = self.get_scalar_type(ctx)
        if self.tiled:
            tile_shape = self.get_tile_shape()
            elem_type = ttcore.ir.TileType.get(
                ctx, tile_shape[0], tile_shape[1], self.dtype
            )
        return elem_type

    def get_device_shape(self, ctx, grid_shape):
        layout = self.build_metal_layout(ctx)
        metal_layout = ttcore.ir.MetalLayoutAttr.maybe_downcast(layout)
        return metal_layout.getDeviceShape(grid_shape, self.get_tile_shape())

    def build_host_tensor_type(self, ctx):
        return RankedTensorType.get(self.logical_shape, self.get_host_elem_type(ctx))

    def build_metal_layout(self, ctx):
        if self._cached_layout is not None:
            return self._cached_layout

        if self.collapse:
            self._cached_layout = ttcore.ir.MetalLayoutAttr.get(
                ctx,
                list(self.logical_shape),
                int(self.mem_space),
                int(ttcore.TensorMemoryLayout.Sharded),
            )
        else:
            empty_interval_type = RankedTensorType.get(
                [0, 2], IntegerType.get_signless(64)
            )
            empty_collapse_intervals = DenseIntElementsAttr.get(empty_interval_type, [])
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
        return RankedTensorType.get(device_shape, elem_type, encoding=layout)

    def build_to_device(self, ctx, val):
        output_type = self.build_device_tensor_type(ctx)
        output = d2m.empty(output_type)
        res = d2m.ToLayoutOp([output_type], val, output).result
        return self.build_blocked_view(ctx, res)

    def build_blocked_view(self, ctx, val):
        if self.blocked_grid_shape == self.grid_shape:
            return val
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
        device_shape = self.get_device_shape(ctx, self.grid_shape)
        blocked_device_shape = self.get_device_shape(ctx, self.blocked_grid_shape)
        device_type = self.build_device_tensor_type(ctx, blocked=False)
        reblock_map = d2m.ir.calculate_reblock_map(
            blocked_device_shape, device_shape, ctx
        )
        return d2m.ViewLayoutOp(device_type, val, reblock_map).result

    def build_from_device(self, ctx, val):
        output_type = self.build_host_tensor_type(ctx)
        output = d2m.empty(output_type)
        return d2m.ToLayoutOp([output_type], val, output).result
