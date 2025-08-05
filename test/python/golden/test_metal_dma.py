# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import Callable, List

from ttir_builder.utils import (
    compile_to_flatbuffer,
    Marks,
    shape_str,
    build_mlir_module,
)
from ttir_builder import Operand, TTIRBuilder, UnitAttr, Shape, TypeInfo
from ttmlir.dialects import ttir, ttcore
from ttmlir.ir import *


def compile_dma_test(test_func, shape, request, write_mlir_to_file=None):

    if write_mlir_to_file is not None:
        with open(write_mlir_to_file, "w") as f:
            f.write(str(build_mlir_module(test_func, [shape])[0]))

    # Back to back tolayout ops are normally folded during canonicalization into
    # a single ToLayoutOp representing the final result. The option
    # 'disable-tolayout-folding' prevents this
    pipeline_options = "{disable-tolayout-folding=1}"
    pipeline = ",".join(
        [
            "ttir-lower-to-layout",
            f"ttir-to-ttmetal-me-pipeline{pipeline_options}",
            f"ttir-to-ttmetal-be-pipeline{pipeline_options}",
        ]
    )
    compile_to_flatbuffer(
        test_func,
        [shape],
        target="ttmetal",
        custom_pipeline=pipeline,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize("shape", [(256, 256)])
@pytest.mark.parametrize("start_grid", [(1, 4), (4, 1), (2, 4), (4, 2)])
@pytest.mark.parametrize("end_grid", [(1, 1), (4, 4), (2, 4), (4, 2)])
@pytest.mark.parametrize(
    "memory_space", [ttcore.MemorySpace.DeviceL1, ttcore.MemorySpace.DeviceDRAM]
)
def test_roundtrip_dma_tiled(
    shape: Shape,
    start_grid: tuple[int, int],
    end_grid: tuple[int, int],
    memory_space: ttcore.MemorySpace,
    request,
):
    def tilize(
        in0: Operand,
        builder: TTIRBuilder,
        unit_attrs: List[str] = None,
    ):
        # derive sharded shapes
        assert (
            (shape[0] % start_grid[0] == 0) and (shape[1] % start_grid[1] == 0),
            "shape must be divisible by start_grid",
        )
        start_shard_shape = (shape[0] // start_grid[0], shape[1] // start_grid[1])
        assert (
            (shape[0] % end_grid[0] == 0) and (shape[1] % end_grid[1] == 0),
            "shard_shape must be divisible by end_grid",
        )
        end_shard_shape = (shape[0] // end_grid[0], shape[1] // end_grid[1])

        # tilize the tensor on a single worker
        to_device = builder.tilize(
            in0,
            output_type=builder.metal_tensor_layout(
                shape,
                tiled=True,
                memorySpace=ttcore.MemorySpace.DeviceL1,
            ),
            unit_attrs=unit_attrs,
        )

        # WRITE tensor from L1 to initial shard layout
        tensor_layoutA = builder.to_layout(
            to_device,
            output_type=builder.metal_tensor_layout(
                start_shard_shape,
                tiled=True,
                memorySpace=memory_space,
                grid=start_grid,
            ),
            unit_attrs=unit_attrs,
        )

        # READ sharded layout to final sharded layout
        tensor_layoutB = builder.to_layout(
            tensor_layoutA,
            output_type=builder.metal_tensor_layout(
                end_shard_shape,
                tiled=True,
                memorySpace=ttcore.MemorySpace.DeviceL1,
                grid=end_grid,
            ),
            unit_attrs=unit_attrs,
        )

        untilize_out = builder.untilize(
            tensor_layoutB,
            output_type=in0.type,
            unit_attrs=unit_attrs,
        )

        return untilize_out

    compile_dma_test(
        tilize,
        shape,
        request,
    )


@pytest.mark.parametrize(
    "shape",
    [(256, 256)],
)
@pytest.mark.parametrize("start_grid", [(1, 1), (1, 2), (2, 1), (4, 4)])
@pytest.mark.parametrize("end_grid", [(1, 1), (1, 2), (2, 1), (4, 4)])
@pytest.mark.parametrize(
    "memory_space", [ttcore.MemorySpace.DeviceL1, ttcore.MemorySpace.DeviceDRAM]
)
def test_roundtrip_dma_rowmajor(
    shape: Shape,
    start_grid: tuple[int, int],
    end_grid: tuple[int, int],
    memory_space: ttcore.MemorySpace,
    request,
):
    def dram_write(
        in0: Operand,
        builder: TTIRBuilder,
        unit_attrs: List[str] = None,
    ):

        to_device = builder.to_layout(
            in0,
            output_type=builder.metal_tensor_layout(
                shape, tiled=False, memorySpace=ttcore.MemorySpace.DeviceL1
            ),
            unit_attrs=unit_attrs,
        )

        # derive sharded shapes
        assert (
            (shape[0] % start_grid[0] == 0) and (shape[1] % start_grid[1] == 0),
            "shape must be divisible by grid",
        )
        start_shard_shape = (shape[0] // start_grid[0], shape[1] // start_grid[1])

        assert (
            (shape[0] % end_grid[0] == 0) and (shape[1] % end_grid[1] == 0),
            "start_shard_shape must be divisible by end_grid",
        )
        end_shard_shape = (shape[0] // end_grid[0], shape[1] // end_grid[1])

        # WRITE L1 to initial shard layout
        tensor_layoutA = builder.to_layout(
            to_device,
            output_type=builder.metal_tensor_layout(
                start_shard_shape,
                tiled=False,
                memorySpace=memory_space,
                grid=start_grid,
            ),
            unit_attrs=unit_attrs,
        )

        # READ sharded layout to final sharded layout
        tensor_layoutB = builder.to_layout(
            tensor_layoutA,
            output_type=builder.metal_tensor_layout(
                end_shard_shape,
                tiled=False,
                memorySpace=ttcore.MemorySpace.DeviceL1,
                grid=end_grid,
            ),
            unit_attrs=unit_attrs,
        )

        system_out = builder.to_layout(
            tensor_layoutB,
            output_type=in0.type,
            unit_attrs=unit_attrs,
        )

        return system_out

    compile_dma_test(dram_write, shape, request)
