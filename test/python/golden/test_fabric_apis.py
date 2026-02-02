# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from collections import OrderedDict
import os
import _ttmlir_runtime as tt_runtime
from ttmlir.dialects import ttcore
from ttmlir.ir import *
from builder.base.builder_runtime import execute_fb
from builder.base.builder_apis import create_custom_ttir_pipeline_fn
from builder.base.builder_utils import run_ttir_pipeline
from golden import GoldenMapTensor
from ttmlir.passes import (
    ttmetal_to_flatbuffer_bin,
    ttnn_to_flatbuffer_bin,
)
from ttmlir.passmanager import PassManager
import math


@pytest.mark.frontend("ttir")
@pytest.mark.parametrize(
    "fabric_config",
    [
        tt_runtime.runtime.FabricConfig.FABRIC_1D,
        tt_runtime.runtime.FabricConfig.FABRIC_1D_RING,
        tt_runtime.runtime.FabricConfig.FABRIC_2D,
        # T3K physically does not support torus fabric configs:
        # tt_runtime.runtime.FabricConfig.FABRIC_2D_TORUS_X,
        # tt_runtime.runtime.FabricConfig.FABRIC_2D_TORUS_Y,
        # tt_runtime.runtime.FabricConfig.FABRIC_2D_TORUS_XY,
    ],
)
@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize("mesh_shape", [(2, 4)])
def test_fabric_p2p(target: str, request, mesh_shape, fabric_config, device):
    with open(
        os.path.join(
            os.path.dirname(__file__), "fabric_api_snippets/test_fabric_p2p.mlir"
        ),
        "r",
        encoding="utf-8",
    ) as f:
        mlir_text = f.read()

    ctx = Context()
    loc = Location.unknown(ctx)

    with ctx, loc:
        module = Module.parse(mlir_text)
        print("Module:", module)

    full_shape = (256, 768)
    input_tensor = torch.zeros(full_shape, dtype=torch.bfloat16)
    input_tensor[0:128, 0:192] = 1.0  # device 0
    input_tensor[0:128, 192:384] = 2.0  # device 1
    input_tensor[0:128, 384:576] = 3.0  # device 2
    input_tensor[0:128, 576:768] = 4.0  # device 3
    input_tensor[128:256, 0:192] = 5.0  # device 4
    input_tensor[128:256, 192:384] = 6.0  # device 5
    input_tensor[128:256, 384:576] = 7.0  # device 6
    input_tensor[128:256, 576:768] = 8.0  # device 7

    # Expected output: device 1's region gets device 0's value (1.0)
    output_tensor = input_tensor.clone()
    output_tensor[0:128, 192:384] = 1.0  # device 1 now has device 0's data

    golden_input_output_tensors = {}
    golden_input_output_tensors[0] = {
        "input_0": GoldenMapTensor({0: input_tensor}, (1, 1)),
        "output_0": GoldenMapTensor({0: output_tensor}, (1, 1)),
    }

    module = run_ttir_pipeline(
        module,
        create_custom_ttir_pipeline_fn(f"ttir-to-ttmetal-pipeline"),
        pipeline_options=[],
        save_artifacts=True,
        system_desc_path=request.config.getoption("--sys-desc"),
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
    )

    _, output_tensors = execute_fb(
        ttmetal_to_flatbuffer_bin(module),
        golden_input_output_tensors,
        {},
        device=device,
        check_pcc=True,
        check_atol=True,
    )


def get_device_id_t3k_1d_mesh_shape(mesh_coord):
    if mesh_coord[1] < 4:
        return mesh_coord[1]
    else:
        return 11 - mesh_coord[1]


def get_device_id_t3k_2d_mesh_shape(mesh_coord):
    return mesh_coord[0] * 4 + mesh_coord[1]


def get_device_id_6u_2d_mesh_shape(mesh_coord):
    return mesh_coord[0] * 4 + mesh_coord[1]


def wrap_range(start, end, size):
    length = (end - start) % size + 1
    return [(start + i) % size for i in range(length)]


@pytest.mark.frontend("ttir")
@pytest.mark.parametrize("fabric_config", [tt_runtime.runtime.FabricConfig.FABRIC_1D])
@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize("mesh_shape", [(1, 8)])
@pytest.mark.parametrize(
    "topology, cluster_axis, routing_mode", [("linear", 1, "bidir_line_mesh")]
)
@pytest.mark.parametrize(
    "src_coord, dst_coord_start, dst_coord_end",
    [
        ((0, 0), (0, 1), (0, 4)),  # right
        ((0, 0), (0, 3), (0, 7)),  # right
        ((0, 5), (0, 1), (0, 3)),  # left
        ((0, 5), (0, 7), (0, 0)),  # wrap
        ((0, 5), (0, 0), (0, 7)),  # in range (broadcast)
        ((0, 3), (0, 0), (0, 5)),  # in range
        ((0, 5), (0, 4), (0, 3)),  # in range, wrap
        # ((0, 5), (0, 4), (0, 1)), # unsupported (has a gap)
    ],
)
def test_fabric_mcast_1x8_line(
    target: str,
    request,
    mesh_shape,
    fabric_config,
    device,
    src_coord,
    dst_coord_start,
    dst_coord_end,
    topology,
    cluster_axis,
    routing_mode,
):
    shard_shape = (32, 32)
    full_shape = (shard_shape[0] * mesh_shape[0], shard_shape[1] * mesh_shape[1])

    with open(
        os.path.join(
            os.path.dirname(__file__), "fabric_api_snippets/test_fabric_mcast.mlir"
        ),
        "r",
        encoding="utf-8",
    ) as f:
        mlir_text = f.read()

    # Replace device IDs with parameterized values
    mlir_text = (
        mlir_text.replace(
            "insert_src_dev_id",
            f"{get_device_id_t3k_1d_mesh_shape(src_coord)}",
        )
        .replace(
            "insert_dst_dev_id_start",
            f"{get_device_id_t3k_1d_mesh_shape(dst_coord_start)}",
        )
        .replace(
            "insert_dst_dev_id_end",
            f"{get_device_id_t3k_1d_mesh_shape(dst_coord_end)}",
        )
        .replace(
            "insert_mesh_shape_0",
            f"{mesh_shape[0]}",
        )
        .replace(
            "insert_mesh_shape_1",
            f"{mesh_shape[1]}",
        )
        .replace(
            "insert_chip_ids",
            f"[{', '.join(f'{i}' for i in range(0, mesh_shape[0] * mesh_shape[1]))}]",
        )
        .replace(
            "insert_full_tensor_shape_0",
            f"{full_shape[0]}",
        )
        .replace(
            "insert_full_tensor_shape_1",
            f"{full_shape[1]}",
        )
        .replace(
            "insert_topology",
            f"{topology}",
        )
        .replace(
            "insert_cluster_axis",
            f"{cluster_axis}",
        )
        .replace(
            "insert_routing_mode",
            f"{routing_mode}",
        )
    )

    ctx = Context()
    loc = Location.unknown(ctx)

    with ctx, loc:
        module = Module.parse(mlir_text)
        print("Module:", module)

    input_tensor = torch.zeros(full_shape, dtype=torch.bfloat16)
    for i in range(0, mesh_shape[0]):
        for j in range(0, mesh_shape[1]):
            start_y = i * shard_shape[0]
            start_x = j * shard_shape[1]
            input_tensor[
                start_y : start_y + shard_shape[0], start_x : start_x + shard_shape[1]
            ] = (i * mesh_shape[1] + j + 1.0)

    # Expected output: devices in dst range get value of src device
    output_tensor = input_tensor.clone()
    srd_device_val = input_tensor[
        src_coord[0] * shard_shape[0], src_coord[1] * shard_shape[1]
    ]
    for i in wrap_range(dst_coord_start[0], dst_coord_end[0], mesh_shape[0]):
        for j in wrap_range(dst_coord_start[1], dst_coord_end[1], mesh_shape[1]):
            start_y = i * shard_shape[0]
            start_x = j * shard_shape[1]
            output_tensor[
                start_y : start_y + shard_shape[0], start_x : start_x + shard_shape[1]
            ] = srd_device_val

    golden_input_output_tensors = {}
    golden_input_output_tensors[0] = {
        "input_0": GoldenMapTensor({0: input_tensor}, (1, 1)),
        "output_0": GoldenMapTensor({0: output_tensor}, (1, 1)),
    }

    module = run_ttir_pipeline(
        module,
        create_custom_ttir_pipeline_fn(f"ttir-to-ttmetal-pipeline"),
        pipeline_options=[],
        save_artifacts=True,
        system_desc_path=request.config.getoption("--sys-desc"),
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
    )

    _, output_tensors = execute_fb(
        ttmetal_to_flatbuffer_bin(module),
        golden_input_output_tensors,
        {},
        device=device,
        check_pcc=True,
        check_atol=True,
    )

    # print unique value in each shard
    for i in range(0, mesh_shape[0]):
        for j in range(0, mesh_shape[1]):
            start_y = i * shard_shape[0]
            start_x = j * shard_shape[1]
            print(
                f"Shard {i * mesh_shape[1] + j}: {output_tensors['program_0']['device_output_0'][start_y:start_y+shard_shape[0], start_x:start_x+shard_shape[1]].unique()}"
            )


@pytest.mark.frontend("ttir")
@pytest.mark.parametrize(
    "fabric_config", [tt_runtime.runtime.FabricConfig.FABRIC_1D_RING]
)
@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize("mesh_shape", [(1, 8)])
@pytest.mark.parametrize(
    "topology, cluster_axis, routing_mode", [("ring", 1, "unidir_ring_torus")]
)
@pytest.mark.parametrize(
    "src_coord, dst_coord_start, dst_coord_end",
    [
        ((0, 0), (0, 1), (0, 4)),
        ((0, 0), (0, 3), (0, 7)),
        ((0, 5), (0, 1), (0, 3)),
        ((0, 5), (0, 0), (0, 7)),
        # ((0, 3), (0, 0), (0, 5)), # unsupported (has a gap)
    ],
)
def test_fabric_mcast_1x8_ring(
    target: str,
    request,
    mesh_shape,
    fabric_config,
    device,
    src_coord,
    dst_coord_start,
    dst_coord_end,
    topology,
    cluster_axis,
    routing_mode,
):
    shard_shape = (32, 32)
    full_shape = (shard_shape[0] * mesh_shape[0], shard_shape[1] * mesh_shape[1])

    with open(
        os.path.join(
            os.path.dirname(__file__), "fabric_api_snippets/test_fabric_mcast.mlir"
        ),
        "r",
        encoding="utf-8",
    ) as f:
        mlir_text = f.read()

    # Replace device IDs with parameterized values
    mlir_text = (
        mlir_text.replace(
            "insert_src_dev_id",
            f"{get_device_id_t3k_1d_mesh_shape(src_coord)}",
        )
        .replace(
            "insert_dst_dev_id_start",
            f"{get_device_id_t3k_1d_mesh_shape(dst_coord_start)}",
        )
        .replace(
            "insert_dst_dev_id_end",
            f"{get_device_id_t3k_1d_mesh_shape(dst_coord_end)}",
        )
        .replace(
            "insert_mesh_shape_0",
            f"{mesh_shape[0]}",
        )
        .replace(
            "insert_mesh_shape_1",
            f"{mesh_shape[1]}",
        )
        .replace(
            "insert_chip_ids",
            f"[{', '.join(f'{i}' for i in range(0, mesh_shape[0] * mesh_shape[1]))}]",
        )
        .replace(
            "insert_full_tensor_shape_0",
            f"{full_shape[0]}",
        )
        .replace(
            "insert_full_tensor_shape_1",
            f"{full_shape[1]}",
        )
        .replace(
            "insert_topology",
            f"{topology}",
        )
        .replace(
            "insert_cluster_axis",
            f"{cluster_axis}",
        )
        .replace(
            "insert_routing_mode",
            f"{routing_mode}",
        )
    )

    ctx = Context()
    loc = Location.unknown(ctx)

    with ctx, loc:
        module = Module.parse(mlir_text)
        print("Module:", module)

    shard_shape = (32, 32)
    full_shape = (shard_shape[0] * mesh_shape[0], shard_shape[1] * mesh_shape[1])
    input_tensor = torch.zeros(full_shape, dtype=torch.bfloat16)
    for i in range(0, mesh_shape[0]):
        for j in range(0, mesh_shape[1]):
            start_y = i * shard_shape[0]
            start_x = j * shard_shape[1]
            input_tensor[
                start_y : start_y + shard_shape[0], start_x : start_x + shard_shape[1]
            ] = (i * mesh_shape[1] + j + 1.0)

    # Expected output: devices in dst range get value of src device
    output_tensor = input_tensor.clone()
    srd_device_val = input_tensor[
        src_coord[0] * shard_shape[0], src_coord[1] * shard_shape[1]
    ]
    for i in wrap_range(dst_coord_start[0], dst_coord_end[0], mesh_shape[0]):
        for j in wrap_range(dst_coord_start[1], dst_coord_end[1], mesh_shape[1]):
            start_y = i * shard_shape[0]
            start_x = j * shard_shape[1]
            output_tensor[
                start_y : start_y + shard_shape[0], start_x : start_x + shard_shape[1]
            ] = srd_device_val

    golden_input_output_tensors = {}
    golden_input_output_tensors[0] = {
        "input_0": GoldenMapTensor({0: input_tensor}, (1, 1)),
        "output_0": GoldenMapTensor({0: output_tensor}, (1, 1)),
    }

    module = run_ttir_pipeline(
        module,
        create_custom_ttir_pipeline_fn(f"ttir-to-ttmetal-pipeline"),
        pipeline_options=[],
        save_artifacts=True,
        system_desc_path=request.config.getoption("--sys-desc"),
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
    )

    _, output_tensors = execute_fb(
        ttmetal_to_flatbuffer_bin(module),
        golden_input_output_tensors,
        {},
        device=device,
        check_pcc=True,
        check_atol=True,
    )

    # print unique value in each shard
    for i in range(0, mesh_shape[0]):
        for j in range(0, mesh_shape[1]):
            start_y = i * shard_shape[0]
            start_x = j * shard_shape[1]
            print(
                f"Shard {i * mesh_shape[1] + j}: {output_tensors['program_0']['device_output_0'][start_y:start_y+shard_shape[0], start_x:start_x+shard_shape[1]].unique()}"
            )


@pytest.mark.frontend("ttir")
@pytest.mark.parametrize("fabric_config", [tt_runtime.runtime.FabricConfig.FABRIC_1D])
@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize("mesh_shape", [(2, 4)])
@pytest.mark.parametrize(
    "topology, cluster_axis, routing_mode", [("linear", 1, "bidir_line_mesh")]
)
@pytest.mark.parametrize(
    "src_coord, dst_coord_start, dst_coord_end",
    [
        ((1, 0), (1, 1), (1, 3)),
        ((0, 3), (0, 0), (0, 1)),
        ((0, 1), (0, 0), (0, 3)),
        # ((1, 3), (1, 2), (1, 0)), # unsupported (has a gap)
    ],
)
def test_fabric_mcast_2x4_line(
    target: str,
    request,
    mesh_shape,
    fabric_config,
    device,
    src_coord,
    dst_coord_start,
    dst_coord_end,
    topology,
    cluster_axis,
    routing_mode,
):
    shard_shape = (32, 32)
    full_shape = (shard_shape[0] * mesh_shape[0], shard_shape[1] * mesh_shape[1])

    with open(
        os.path.join(
            os.path.dirname(__file__), "fabric_api_snippets/test_fabric_mcast.mlir"
        ),
        "r",
        encoding="utf-8",
    ) as f:
        mlir_text = f.read()

    # Replace device IDs with parameterized values
    mlir_text = (
        mlir_text.replace(
            "insert_src_dev_id",
            f"{get_device_id_t3k_2d_mesh_shape(src_coord)}",
        )
        .replace(
            "insert_dst_dev_id_start",
            f"{get_device_id_t3k_2d_mesh_shape(dst_coord_start)}",
        )
        .replace(
            "insert_dst_dev_id_end",
            f"{get_device_id_t3k_2d_mesh_shape(dst_coord_end)}",
        )
        .replace(
            "insert_mesh_shape_0",
            f"{mesh_shape[0]}",
        )
        .replace(
            "insert_mesh_shape_1",
            f"{mesh_shape[1]}",
        )
        .replace(
            "insert_chip_ids",
            f"[{', '.join(f'{i}' for i in range(0, mesh_shape[0] * mesh_shape[1]))}]",
        )
        .replace(
            "insert_full_tensor_shape_0",
            f"{full_shape[0]}",
        )
        .replace(
            "insert_full_tensor_shape_1",
            f"{full_shape[1]}",
        )
        .replace(
            "insert_topology",
            f"{topology}",
        )
        .replace(
            "insert_cluster_axis",
            f"{cluster_axis}",
        )
        .replace(
            "insert_routing_mode",
            f"{routing_mode}",
        )
    )

    ctx = Context()
    loc = Location.unknown(ctx)

    with ctx, loc:
        module = Module.parse(mlir_text)
        print("Module:", module)

    input_tensor = torch.zeros(full_shape, dtype=torch.bfloat16)
    for i in range(0, mesh_shape[0]):
        for j in range(0, mesh_shape[1]):
            start_y = i * shard_shape[0]
            start_x = j * shard_shape[1]
            input_tensor[
                start_y : start_y + shard_shape[0], start_x : start_x + shard_shape[1]
            ] = (i * mesh_shape[1] + j + 1.0)

    # Expected output: devices in dst range get value of src device
    output_tensor = input_tensor.clone()
    srd_device_val = input_tensor[
        src_coord[0] * shard_shape[0], src_coord[1] * shard_shape[1]
    ]
    for i in range(dst_coord_start[0], dst_coord_end[0] + 1):
        for j in range(dst_coord_start[1], dst_coord_end[1] + 1):
            start_y = i * shard_shape[0]
            start_x = j * shard_shape[1]
            output_tensor[
                start_y : start_y + shard_shape[0], start_x : start_x + shard_shape[1]
            ] = srd_device_val

    golden_input_output_tensors = {}
    golden_input_output_tensors[0] = {
        "input_0": GoldenMapTensor({0: input_tensor}, (1, 1)),
        "output_0": GoldenMapTensor({0: output_tensor}, (1, 1)),
    }

    module = run_ttir_pipeline(
        module,
        create_custom_ttir_pipeline_fn(f"ttir-to-ttmetal-pipeline"),
        pipeline_options=[],
        save_artifacts=True,
        system_desc_path=request.config.getoption("--sys-desc"),
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
    )

    _, output_tensors = execute_fb(
        ttmetal_to_flatbuffer_bin(module),
        golden_input_output_tensors,
        {},
        device=device,
        check_pcc=True,
        check_atol=True,
    )

    # print unique value in each shard
    for i in range(0, mesh_shape[0]):
        for j in range(0, mesh_shape[1]):
            start_y = i * shard_shape[0]
            start_x = j * shard_shape[1]
            print(
                f"Shard {i * mesh_shape[1] + j}: {output_tensors['program_0']['device_output_0'][start_y:start_y+shard_shape[0], start_x:start_x+shard_shape[1]].unique()}"
            )


# TODO:Issue if we use 4x8 since the directions are inverted and apis currently assume ns is dim 0 (look into if this is fixable)
@pytest.mark.frontend("ttir")
@pytest.mark.parametrize(
    "fabric_config", [tt_runtime.runtime.FabricConfig.FABRIC_2D_TORUS_XY]
)
@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize("mesh_shape", [(8, 4)])
@pytest.mark.parametrize(
    "topology, cluster_axis, routing_mode", [("ring", 1, "unidir_ring_torus")]
)
@pytest.mark.parametrize(
    "src_coord, dst_coord_start, dst_coord_end",
    [
        ((0, 0), (0, 1), (0, 3)),
        ((0, 3), (0, 0), (0, 1)),
        ((0, 1), (0, 0), (0, 3)),
        # ((0, 0), (0, 3), (0, 7)), # start dist not supported in 2d fabric mcast
        # ((0, 3), (0, 0), (0, 5)), # unsupported (has a gap)
    ],
)
def test_fabric_mcast_8x4_ring(
    target: str,
    request,
    mesh_shape,
    fabric_config,
    device,
    src_coord,
    dst_coord_start,
    dst_coord_end,
    topology,
    cluster_axis,
    routing_mode,
):
    shard_shape = (32, 32)
    full_shape = (shard_shape[0] * mesh_shape[0], shard_shape[1] * mesh_shape[1])

    with open(
        os.path.join(
            os.path.dirname(__file__), "fabric_api_snippets/test_fabric_mcast.mlir"
        ),
        "r",
        encoding="utf-8",
    ) as f:
        mlir_text = f.read()

    # Replace device IDs with parameterized values
    mlir_text = (
        mlir_text.replace(
            "insert_src_dev_id",
            f"{get_device_id_6u_2d_mesh_shape(src_coord)}",
        )
        .replace(
            "insert_dst_dev_id_start",
            f"{get_device_id_6u_2d_mesh_shape(dst_coord_start)}",
        )
        .replace(
            "insert_dst_dev_id_end",
            f"{get_device_id_6u_2d_mesh_shape(dst_coord_end)}",
        )
        .replace(
            "insert_mesh_shape_0",
            f"{mesh_shape[0]}",
        )
        .replace(
            "insert_mesh_shape_1",
            f"{mesh_shape[1]}",
        )
        .replace(
            "insert_chip_ids",
            f"[{', '.join(f'{i}' for i in range(0, mesh_shape[0] * mesh_shape[1]))}]",
        )
        .replace(
            "insert_full_tensor_shape_0",
            f"{full_shape[0]}",
        )
        .replace(
            "insert_full_tensor_shape_1",
            f"{full_shape[1]}",
        )
        .replace(
            "insert_topology",
            f"{topology}",
        )
        .replace(
            "insert_cluster_axis",
            f"{cluster_axis}",
        )
        .replace(
            "insert_routing_mode",
            f"{routing_mode}",
        )
    )

    ctx = Context()
    loc = Location.unknown(ctx)

    with ctx, loc:
        module = Module.parse(mlir_text)
        print("Module:", module)

    input_tensor = torch.zeros(full_shape, dtype=torch.bfloat16)
    for i in range(0, mesh_shape[0]):
        for j in range(0, mesh_shape[1]):
            start_y = i * shard_shape[0]
            start_x = j * shard_shape[1]
            input_tensor[
                start_y : start_y + shard_shape[0], start_x : start_x + shard_shape[1]
            ] = (i * mesh_shape[1] + j + 1.0)

    # Expected output: devices in dst range get value of src device
    output_tensor = input_tensor.clone()
    srd_device_val = input_tensor[
        src_coord[0] * shard_shape[0], src_coord[1] * shard_shape[1]
    ]
    for i in wrap_range(dst_coord_start[0], dst_coord_end[0], mesh_shape[0]):
        for j in wrap_range(dst_coord_start[1], dst_coord_end[1], mesh_shape[1]):
            start_y = i * shard_shape[0]
            start_x = j * shard_shape[1]
            output_tensor[
                start_y : start_y + shard_shape[0], start_x : start_x + shard_shape[1]
            ] = srd_device_val

    golden_input_output_tensors = {}
    golden_input_output_tensors[0] = {
        "input_0": GoldenMapTensor({0: input_tensor}, (1, 1)),
        "output_0": GoldenMapTensor({0: output_tensor}, (1, 1)),
    }

    module = run_ttir_pipeline(
        module,
        create_custom_ttir_pipeline_fn(f"ttir-to-ttmetal-pipeline"),
        pipeline_options=[],
        save_artifacts=True,
        system_desc_path=request.config.getoption("--sys-desc"),
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
    )

    _, output_tensors = execute_fb(
        ttmetal_to_flatbuffer_bin(module),
        golden_input_output_tensors,
        {},
        device=device,
        check_pcc=True,
        check_atol=True,
    )

    # print unique value in each shard
    for i in range(0, mesh_shape[0]):
        for j in range(0, mesh_shape[1]):
            start_y = i * shard_shape[0]
            start_x = j * shard_shape[1]
            print(
                f"Shard {i * mesh_shape[1] + j}: {output_tensors['program_0']['device_output_0'][start_y:start_y+shard_shape[0], start_x:start_x+shard_shape[1]].unique()}"
            )


# TODO:Issue if we use 4x8 since the directions are inverted and apis currently assume ns is dim 0 (look into if this is fixable)
@pytest.mark.frontend("ttir")
@pytest.mark.parametrize(
    "fabric_config", [tt_runtime.runtime.FabricConfig.FABRIC_2D_TORUS_XY]
)
@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize("mesh_shape", [(8, 4)])
@pytest.mark.parametrize(
    "topology, cluster_axis, routing_mode", [("torus", 1, "unidir_ring_torus")]
)
@pytest.mark.parametrize(
    "src_coord, dst_coord_start, dst_coord_end",
    [
        ((0, 0), (0, 1), (0, 3)),  # dim 1
        ((0, 1), (0, 0), (0, 3)),  # dim 1
        ((5, 2), (6, 2), (7, 2)),  # dim 0
        ((3, 3), (0, 3), (7, 3)),  # dim 0
        ((1, 1), (2, 1), (4, 3)),  # dim 0, 1
        ((1, 1), (1, 1), (4, 3)),  # dim 0, 1
        ((3, 2), (0, 0), (7, 3)),  # broadcast
        # ((0, 0), (0, 3), (0, 7)), # start dist not supported in 2d fabric mcast
        # ((0, 3), (0, 0), (0, 5)), # unsupported (has a gap)
    ],
)
def test_fabric_mcast_8x4_torus(
    target: str,
    request,
    mesh_shape,
    fabric_config,
    device,
    src_coord,
    dst_coord_start,
    dst_coord_end,
    topology,
    cluster_axis,
    routing_mode,
):
    shard_shape = (32, 32)
    full_shape = (shard_shape[0] * mesh_shape[0], shard_shape[1] * mesh_shape[1])

    with open(
        os.path.join(
            os.path.dirname(__file__), "fabric_api_snippets/test_fabric_mcast.mlir"
        ),
        "r",
        encoding="utf-8",
    ) as f:
        mlir_text = f.read()

    # Replace device IDs with parameterized values
    mlir_text = (
        mlir_text.replace(
            "insert_src_dev_id",
            f"{get_device_id_6u_2d_mesh_shape(src_coord)}",
        )
        .replace(
            "insert_dst_dev_id_start",
            f"{get_device_id_6u_2d_mesh_shape(dst_coord_start)}",
        )
        .replace(
            "insert_dst_dev_id_end",
            f"{get_device_id_6u_2d_mesh_shape(dst_coord_end)}",
        )
        .replace(
            "insert_mesh_shape_0",
            f"{mesh_shape[0]}",
        )
        .replace(
            "insert_mesh_shape_1",
            f"{mesh_shape[1]}",
        )
        .replace(
            "insert_chip_ids",
            f"[{', '.join(f'{i}' for i in range(0, mesh_shape[0] * mesh_shape[1]))}]",
        )
        .replace(
            "insert_full_tensor_shape_0",
            f"{full_shape[0]}",
        )
        .replace(
            "insert_full_tensor_shape_1",
            f"{full_shape[1]}",
        )
        .replace(
            "insert_topology",
            f"{topology}",
        )
        .replace(
            "insert_cluster_axis",
            f"{cluster_axis}",
        )
        .replace(
            "insert_routing_mode",
            f"{routing_mode}",
        )
    )

    ctx = Context()
    loc = Location.unknown(ctx)

    with ctx, loc:
        module = Module.parse(mlir_text)
        print("Module:", module)

    input_tensor = torch.zeros(full_shape, dtype=torch.bfloat16)
    for i in range(0, mesh_shape[0]):
        for j in range(0, mesh_shape[1]):
            start_y = i * shard_shape[0]
            start_x = j * shard_shape[1]
            input_tensor[
                start_y : start_y + shard_shape[0], start_x : start_x + shard_shape[1]
            ] = (i * mesh_shape[1] + j + 1.0)

    # Expected output: devices in dst range get value of src device
    output_tensor = input_tensor.clone()
    srd_device_val = input_tensor[
        src_coord[0] * shard_shape[0], src_coord[1] * shard_shape[1]
    ]
    for i in wrap_range(dst_coord_start[0], dst_coord_end[0], mesh_shape[0]):
        for j in wrap_range(dst_coord_start[1], dst_coord_end[1], mesh_shape[1]):
            start_y = i * shard_shape[0]
            start_x = j * shard_shape[1]
            output_tensor[
                start_y : start_y + shard_shape[0], start_x : start_x + shard_shape[1]
            ] = srd_device_val

    golden_input_output_tensors = {}
    golden_input_output_tensors[0] = {
        "input_0": GoldenMapTensor({0: input_tensor}, (1, 1)),
        "output_0": GoldenMapTensor({0: output_tensor}, (1, 1)),
    }

    module = run_ttir_pipeline(
        module,
        create_custom_ttir_pipeline_fn(f"ttir-to-ttmetal-pipeline"),
        pipeline_options=[],
        save_artifacts=True,
        system_desc_path=request.config.getoption("--sys-desc"),
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
    )

    _, output_tensors = execute_fb(
        ttmetal_to_flatbuffer_bin(module),
        golden_input_output_tensors,
        {},
        device=device,
        check_pcc=True,
        check_atol=True,
    )

    # print unique value in each shard
    for i in range(0, mesh_shape[0]):
        for j in range(0, mesh_shape[1]):
            start_y = i * shard_shape[0]
            start_x = j * shard_shape[1]
            print(
                f"Shard {i * mesh_shape[1] + j}: {output_tensors['program_0']['device_output_0'][start_y:start_y+shard_shape[0], start_x:start_x+shard_shape[1]].unique()}"
            )

@pytest.mark.frontend("ttnn")
@pytest.mark.parametrize(
    "fabric_config",
    [
        tt_runtime.runtime.FabricConfig.FABRIC_1D,
        tt_runtime.runtime.FabricConfig.FABRIC_1D_RING,
        tt_runtime.runtime.FabricConfig.FABRIC_2D,
    ],
)
@pytest.mark.parametrize("target", ["ttnn"])
@pytest.mark.parametrize("mesh_shape", [(2, 4)])
def test_ttnn_generic_p2p(target: str, request, mesh_shape, fabric_config, device):
    with open(
        os.path.join(
            os.path.dirname(__file__), "fabric_api_snippets/test_generic_op_p2p.mlir"
        ),
        "r",
        encoding="utf-8",
    ) as f:
        mlir_text = f.read()

    ctx = Context()
    loc = Location.unknown(ctx)

    with ctx, loc:
        module = Module.parse(mlir_text)
        print("Module:", module)

    full_shape = (64, 128)

    # Input tensor (arg0): source data with distinct values per device.
    input_tensor = torch.zeros(full_shape, dtype=torch.bfloat16)
    input_tensor[0:32, 0:32] = 1.0  # device 0
    input_tensor[0:32, 32:64] = 2.0  # device 1
    input_tensor[0:32, 64:96] = 3.0  # device 2
    input_tensor[0:32, 96:128] = 4.0  # device 3
    input_tensor[32:64, 0:32] = 5.0  # device 4
    input_tensor[32:64, 32:64] = 6.0  # device 5
    input_tensor[32:64, 64:96] = 7.0  # device 6
    input_tensor[32:64, 96:128] = 8.0  # device 7

    output_preallocated = input_tensor.clone()

    # Expected output: device 1's region gets device 0's value of 1.0.
    # All other regions remain 0.
    expected_output = output_preallocated.clone()
    expected_output[0:32, 32:64] = 1.0  # device 1 now has device 0's data

    golden_input_output_tensors = {}
    golden_input_output_tensors[0] = {
        "input_0": GoldenMapTensor({0: input_tensor}, (1, 1)),
        "input_1": GoldenMapTensor({0: output_preallocated}, (1, 1)),
        "output_0": GoldenMapTensor({0: expected_output}, (1, 1)),
    }

    # Register device (add system_desc) before flatbuffer conversion.
    system_desc_path = request.config.getoption("--sys-desc")
    mesh_shape_str = f"{mesh_shape[0]},{mesh_shape[1]}"
    with module.context:
        pm = PassManager.parse(
            f"builtin.module(ttcore-register-device{{system-desc-path={system_desc_path} mesh-shape={mesh_shape_str}}})"
        )
        pm.run(module.operation)

    print(f"module: {module}")

    _, output_tensors = execute_fb(
        ttnn_to_flatbuffer_bin(module),
        golden_input_output_tensors,
        {},
        device=device,
        check_pcc=True,
        check_atol=True,
        check_rtol=True,
    )

    # program_outputs = output_tensors["program_0"]Collapse commentComment on line R171sohaibnadeemTT commented on Jan 28, 2026 sohaibnadeemTTon Jan 28, 2026ContributorMore actionsnit: clean upReactWrite a replyResolve comment
    # actual_output = program_outputs["device_output_0"]

    # print(f"Device 0 region [0:32, 0:32]: = \n{actual_output[0:32, 0:32]}")
    # print(f"Device 1 region [0:32, 32:64]: = \n{actual_output[0:32, 32:64]}")
    # print(f"Device 2 region [0:32, 64:96]: = \n{actual_output[0:32, 64:96]}")
    # print(f"Device 3 region [0:32, 96:128]: = \n{actual_output[0:32, 96:128]}")
    # print(f"Device 4 region [32:64, 0:32]: = \n{actual_output[32:64, 0:32]}")
    # print(f"Device 5 region [32:64, 32:64]: = \n{actual_output[32:64, 32:64]}")
    # print(f"Device 6 region [32:64, 64:96]: = \n{actual_output[32:64, 64:96]}")
    # print(f"Device 7 region [32:64, 96:128]: = \n{actual_output[32:64, 96:128]}"):
