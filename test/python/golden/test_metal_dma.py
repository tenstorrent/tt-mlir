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


@pytest.mark.skip
@pytest.mark.parametrize(
    "shape", [(x, y) for x in (32, 128) for y in (32, 64, 128)] + [(256, 256)]
)
def test_dram_tilize(shape: Shape, request):
    def tilize(
        in0: Operand,
        builder: TTIRBuilder,
        unit_attrs: List[str] = None,
    ):

        to_device = builder.to_layout(
            in0,
            output_type=builder.metal_tensor_layout(
                shape, tiled=False, memorySpace=ttcore.MemorySpace.DeviceDRAM
            ),
            unit_attrs=unit_attrs,
        )

        tilize_out = builder.tilize(
            to_device,
            output_type=builder.metal_tensor_layout(
                shape, tiled=True, memorySpace=ttcore.MemorySpace.DeviceL1
            ),
            unit_attrs=unit_attrs,
        )

        untilize_out = builder.untilize(
            tilize_out,
            output_type=in0.type,
            unit_attrs=unit_attrs,
        )

        return untilize_out

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
        tilize,
        [shape],
        target="ttmetal",
        custom_pipeline=pipeline,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize(
    "shape",
    [(256, 256)],
)
@pytest.mark.parametrize("grid", [(1, 1), (2, 2), (1, 2)])
def test_dram_write(shape: Shape, grid: tuple[int, int], request):
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

        # derive sharded shape for DRAM
        assert (
            (shape[0] % grid[0] == 0) and (shape[1] % grid[1] == 0),
            "shape must be divisible by grid",
        )
        dram_sharded_shape = (shape[0] // grid[0], shape[1] // grid[1])

        # write L1 to DRAM
        to_device_dram = builder.to_layout(
            to_device,
            output_type=builder.metal_tensor_layout(
                dram_sharded_shape,
                tiled=False,
                memorySpace=ttcore.MemorySpace.DeviceDRAM,
                grid=grid,
            ),
            unit_attrs=unit_attrs,
        )
        print("dram_sharded_shape: ", dram_sharded_shape)

        to_device_l1 = builder.to_layout(
            to_device_dram,
            output_type=builder.metal_tensor_layout(
                dram_sharded_shape,
                tiled=False,
                memorySpace=ttcore.MemorySpace.DeviceL1,
                grid=grid,
            ),
            unit_attrs=unit_attrs,
        )

        system_out = builder.to_layout(
            to_device_l1,
            output_type=in0.type,
            unit_attrs=unit_attrs,
        )

        return system_out

    with open(f"test_dram_write_{grid[0]}x{grid[1]}.mlir", "w") as f:
        f.write(str(build_mlir_module(dram_write, [shape])[0]))

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
        dram_write,
        [shape],
        target="ttmetal",
        custom_pipeline=pipeline,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )
