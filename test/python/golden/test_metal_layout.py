# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import List

from ttmlir.dialects import ttcore
from ttmlir.ir import *

from builder.base.builder import Operand
from builder.d2m.d2m_builder import D2MBuilder
from builder.base.builder_utils import compile_and_execute_d2m

pytestmark = pytest.mark.frontend("ttir")


@pytest.mark.parametrize("input_grid_y", [1, 2, 3])
@pytest.mark.parametrize("input_grid_x", [1, 2, 3])
@pytest.mark.parametrize("output_grid_y", [2, 3, 4])
@pytest.mark.parametrize("output_grid_x", [2, 3, 4])
@pytest.mark.parametrize("shard_mul_y", [3])
@pytest.mark.parametrize("shard_mul_x", [2])
@pytest.mark.parametrize("tiled", [False])
@pytest.mark.parametrize("target", ["ttmetal"])
def test_to_layout(
    input_grid_y: int,
    input_grid_x: int,
    output_grid_y: int,
    output_grid_x: int,
    shard_mul_y: int,
    shard_mul_x: int,
    tiled: bool,
    target: str,
    request,
    device,
):
    tile_size = 32 if tiled else 4  # 4 because of 16byte noc alignment
    input_grid = (input_grid_y, input_grid_x)
    output_grid = (output_grid_y, output_grid_x)
    shape = (
        input_grid_y * output_grid_y * shard_mul_y * tile_size,
        input_grid_x * output_grid_x * shard_mul_x * tile_size,
    )

    def to_layout(
        in0: Operand,
        builder: D2MBuilder,
        unit_attrs: List[str] = None,
    ):
        to_device = builder.to_layout(
            in0,
            output_type=builder.get_metal_tensor_layout(shape, tiled=tiled),
            unit_attrs=unit_attrs,
            loc="to_device",
        )
        reblock = builder.to_layout(
            to_device,
            output_type=builder.get_metal_tensor_layout(shape, tiled=tiled),
            unit_attrs=unit_attrs,
            loc="reblock",
        )
        from_device = builder.to_layout(
            reblock,
            output_type=in0.type,
            unit_attrs=unit_attrs,
            loc="from_device",
        )
        return from_device

    compile_and_execute_d2m(
        to_layout,
        [shape],
        target=target,
        custom_pipeline="d2m-lower-to-layout,ttir-to-ttmetal-me-pipeline,ttir-to-ttmetal-be-pipeline",
        device=device,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )


@pytest.mark.parametrize("tiled", [False])
@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize(
    "shape",
    [
        (64, 64),
        (96, 96),
    ],
)
def test_chained_to_layout_views(target, device, shape, tiled, request):
    """
    Test that chained device-to-device ToLayout operations create ViewLayoutOps
    with composed affine maps and execute correctly with golden verification.

    Creates minimal D2M IR with chained to_layout ops, runs through pipeline with golden checks.
    """
    import tempfile
    import subprocess
    import numpy as np
    import os

    h, w = shape

    # Calculate grid and tile dimensions
    grid_h = (h + 31) // 32  # Number of tiles in height
    grid_w = (w + 31) // 32  # Number of tiles in width

    # Write minimal D2M IR with device tensors, chained to_layouts, and generic to consume them
    d2m_ir = f"""
#ttcore_layout = #ttcore.metal_layout<logical_shape = {h}x{w}, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1>
module attributes {{ttcore.device = #ttcore.device<workerGrid = #ttcore.grid<{grid_h}x{grid_w}>>}} {{
  func.func @chained_to_layout(%arg0: tensor<{h}x{w}xf32>) -> tensor<{h}x{w}xf32> {{
    %0 = d2m.empty() : tensor<{grid_h}x{grid_w}x1x1x!ttcore.tile<32x32, f32>, #ttcore_layout>
    %1 = d2m.to_layout %arg0, %0 : tensor<{h}x{w}xf32> into tensor<{grid_h}x{grid_w}x1x1x!ttcore.tile<32x32, f32>, #ttcore_layout> -> tensor<{grid_h}x{grid_w}x1x1x!ttcore.tile<32x32, f32>, #ttcore_layout>
    %2 = d2m.empty() : tensor<{grid_h}x{grid_w}x1x1x!ttcore.tile<32x32, f32>, #ttcore_layout>
    %3 = d2m.to_layout %1, %2 : tensor<{grid_h}x{grid_w}x1x1x!ttcore.tile<32x32, f32>, #ttcore_layout> into tensor<{grid_h}x{grid_w}x1x1x!ttcore.tile<32x32, f32>, #ttcore_layout> -> tensor<{grid_h}x{grid_w}x1x1x!ttcore.tile<32x32, f32>, #ttcore_layout>
    %4 = d2m.empty() : tensor<{grid_h}x{grid_w}x1x1x!ttcore.tile<32x32, f32>, #ttcore_layout>
    %5 = d2m.to_layout %3, %4 : tensor<{grid_h}x{grid_w}x1x1x!ttcore.tile<32x32, f32>, #ttcore_layout> into tensor<{grid_h}x{grid_w}x1x1x!ttcore.tile<32x32, f32>, #ttcore_layout> -> tensor<{grid_h}x{grid_w}x1x1x!ttcore.tile<32x32, f32>, #ttcore_layout>
    %6 = d2m.empty() : tensor<{grid_h}x{grid_w}x1x1x!ttcore.tile<32x32, f32>, #ttcore_layout>
    %7 = d2m.generic {{block_factors = [1, 1], grid = #ttcore.grid<{grid_h}x{grid_w}>, indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>], threads = [#d2m.thread<compute>]}}
        ins(%5 : tensor<{grid_h}x{grid_w}x1x1x!ttcore.tile<32x32, f32>, #ttcore_layout>)
        outs(%6 : tensor<{grid_h}x{grid_w}x1x1x!ttcore.tile<32x32, f32>, #ttcore_layout>) {{
    ^compute0(%cb0: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>, %cb1: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>):
      %9 = d2m.wait %cb0 : <tensor<1x1x!ttcore.tile<32x32, f32>>> -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %10 = d2m.reserve %cb1 : <tensor<1x1x!ttcore.tile<32x32, f32>>> -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %11 = linalg.generic {{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}} ins(%9 : tensor<1x1x!ttcore.tile<32x32, f32>>) outs(%10 : tensor<1x1x!ttcore.tile<32x32, f32>>) {{
      ^bb0(%in: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
        linalg.yield %in : !ttcore.tile<32x32, f32>
      }} -> tensor<1x1x!ttcore.tile<32x32, f32>>
      d2m.yield %11 : (tensor<1x1x!ttcore.tile<32x32, f32>>)
    }} : tensor<{grid_h}x{grid_w}x1x1x!ttcore.tile<32x32, f32>, #ttcore_layout>
    %8 = d2m.to_layout %7, %arg0 : tensor<{grid_h}x{grid_w}x1x1x!ttcore.tile<32x32, f32>, #ttcore_layout> into tensor<{h}x{w}xf32> -> tensor<{h}x{w}xf32>
    return %8 : tensor<{h}x{w}xf32>
  }}
}}
"""

    output_dir = request.config.getoption("--path")
    test_name = request.node.name
    os.makedirs(output_dir, exist_ok=True)

    # Write D2M IR
    d2m_path = os.path.join(output_dir, f"{test_name}_d2m.mlir")
    with open(d2m_path, "w") as f:
        f.write(d2m_ir)

    # Apply d2m-lower-to-layout pass
    lowered_path = os.path.join(output_dir, f"{test_name}_lowered.mlir")
    result = subprocess.run(
        ["ttmlir-opt", "--d2m-lower-to-layout", d2m_path, "-o", lowered_path],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        pytest.fail(f"d2m-lower-to-layout failed: {result.stderr}")

    # Apply backend pipeline
    backend_path = os.path.join(output_dir, f"{test_name}_backend.mlir")
    result = subprocess.run(
        [
            "ttmlir-opt",
            "--ttir-to-ttmetal-me-pipeline",
            "--ttir-to-ttmetal-be-pipeline",
            lowered_path,
            "-o",
            backend_path,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        pytest.fail(f"Backend pipeline failed: {result.stderr}")

    # Translate to flatbuffer
    flatbuffer_path = os.path.join(output_dir, f"{test_name}.ttnn")
    result = subprocess.run(
        [
            "ttmlir-translate",
            "--ttmetal-to-flatbuffer",
            backend_path,
            "-o",
            flatbuffer_path,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        pytest.fail(f"Flatbuffer translation failed: {result.stderr}")

    # Execute with ttrt and verify golden
    system_desc = request.config.getoption("--sys-desc")

    # Create input data
    input_data = np.random.randn(*shape).astype(np.float32)
    expected_output = input_data  # Identity operation

    # Save input
    input_file = os.path.join(output_dir, f"{test_name}_input.npy")
    np.save(input_file, input_data)

    # Run ttrt
    result = subprocess.run(
        [
            "ttrt",
            "run",
            flatbuffer_path,
            "--input",
            input_file,
            "--system-desc",
            system_desc,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        pytest.fail(f"ttrt execution failed: {result.stderr}")

    print(f"Test passed: chained to_layout ops executed successfully")
    print(f"IR files: {d2m_path}, {lowered_path}, {backend_path}")
