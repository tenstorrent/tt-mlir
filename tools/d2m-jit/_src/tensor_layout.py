# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from ttmlir.ir import *
from ttmlir.dialects import ttcore, d2m


class TensorLayout:
    def __init__(
        self,
        tensor,
        block_shape,
        grid_shape=None,
        dtype=None,
        tiled=True,
        collapse=True,
        mem_space=ttcore.MemorySpace.DeviceL1,
    ):
        dtype = TensorLayout._to_data_type(
            str(tensor.dtype if dtype is None else dtype)
        )
        self.tensor = tensor
        self.logical_shape = tensor.shape
        self.block_shape = block_shape
        self.blocked_grid_shape = TensorLayout._derive_blocked_grid_shape(
            list(tensor.shape), block_shape, tiled
        )
        self.grid_shape = self.logical_grid_shape if grid_shape is None else grid_shape
        self.dtype = dtype
        self.tiled = tiled
        self.collapse = collapse
        self.mem_space = TensorLayout._to_mem_space(mem_space)
        self._cached_layout = None

    @staticmethod
    def _to_data_type(dtype: str):
        if dtype in {"torch.float32", "fp32"}:
            return ttcore.DataType.Float32
        elif dtype in {"torch.float16", "fp16"}:
            return ttcore.DataType.Float16
        elif dtype in {"torch.bfloat16", "bf16"}:
            return ttcore.DataType.BFloat16
        else:
            raise TypeError(f"Unsupported dtype {dtype}")

    @staticmethod
    def _to_mem_space(mem_space):
        if isinstance(mem_space, ttcore.MemorySpace):
            return mem_space
        if mem_space in {"l1", "sram"}:
            return ttcore.MemorySpace.DeviceL1
        elif mem_space == "dram":
            return ttcore.MemorySpace.DeviceDRAM
        else:
            raise TypeError(f"Unsupported mem_space {mem_space}")

    @staticmethod
    def _derive_blocked_grid_shape(logical_shape, block_shape, tiled):
        assert len(logical_shape) == len(block_shape)
        if tiled:
            for i in range(len(logical_shape)):
                logical_shape[i] = (logical_shape[i] + 31) // 32

        blocked_grid_shape = []
        for ls, bs in zip(logical_shape, block_shape):
            assert ls % bs == 0
            blocked_grid_shape.append(ls // bs)

        return blocked_grid_shape

    def get_tile_shape(self):
        return [32, 32] if self.tiled else []

    def get_scalar_type(self, ctx):
        if self.dtype == ttcore.DataType.Float32:
            return F32Type.get(ctx)
        elif self.dtype == ttcore.DataType.Float16:
            return F16Type.get(ctx)
        elif self.dtype == ttcore.DataType.BFloat16:
            return BF16Type.get(ctx)
        else:
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
                int(ttcore.OOBVal.Undef),
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
                int(ttcore.OOBVal.Undef),
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
        res = d2m.ToLayoutOp(
            [output_type],
            val,
            output,
        ).result
        return self.build_blocked_view(ctx, res)

    def build_blocked_view(self, ctx, val):
        if self.blocked_grid_shape == self.grid_shape:
            # Nothing to do
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
            # Nothing to do
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
        return d2m.ToLayoutOp(
            [output_type],
            val,
            output,
        ).result
