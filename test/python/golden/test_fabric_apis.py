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
