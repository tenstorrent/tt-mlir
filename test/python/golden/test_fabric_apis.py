# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from collections import OrderedDict
import os
from ttmlir.dialects import ttcore
from ttmlir.ir import *
from builder.base.builder_runtime import execute_fb
from builder.base.builder_apis import create_custom_ttir_pipeline_fn
from builder.base.builder_utils import run_ttir_pipeline
from golden import GoldenMapTensor
from ttmlir.passes import (
    ttmetal_to_flatbuffer_bin,
)

@pytest.mark.parametrize("fabric_config", [tt_runtime.runtime.FabricConfig.FABRIC_2D])
@pytest.mark.parametrize("target", ["ttmetal"])
@pytest.mark.parametrize("mesh_shape", [(2, 4)])
def test_fabric_p2p(target: str, request, mesh_shape, device):
    with open(os.path.join(os.path.dirname(__file__), "fabric_api_snippets/test_fabric_p2p.mlir"), "r", encoding="utf-8") as f:
        mlir_text = f.read()

    ctx = Context()
    loc = Location.unknown(ctx)

    with ctx, loc:
        module = Module.parse(mlir_text)
        print("Module:", module) 
    
    golden_input_output_tensors = {}
    golden_input_output_tensors[0] = {
        "output_0": GoldenMapTensor({0: torch.full((256, 768), 1.0, dtype=torch.bfloat16)}, (1, 1)),
        "input_0": GoldenMapTensor({0: torch.full((256, 768), 1.0, dtype=torch.bfloat16)}, (1, 1)) 
    }

    module = run_ttir_pipeline(
        module,
        create_custom_ttir_pipeline_fn(f"ttir-to-ttmetal-pipeline"),
        pipeline_options=[],
        save_artifacts=True,
        output_file_name="potato.txt",
        system_desc_path=request.config.getoption("--sys-desc"),
        mesh_dict=OrderedDict([("x", mesh_shape[0]), ("y", mesh_shape[1])]),
    )
        
    execute_fb(
        ttmetal_to_flatbuffer_bin(module),
        golden_input_output_tensors,
        {},
        device=device,
    )
  