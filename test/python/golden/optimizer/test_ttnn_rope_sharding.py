# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

import pytest
import torch

from builder.base.builder import Shape
from builder.base.builder_utils import compile_and_execute_ttnn
from builder.ttnn.ttnn_builder import TTNNBuilder

pytestmark = pytest.mark.frontend("ttnn")


@pytest.mark.parametrize(
    "shapes",
    [
        [(1, 1, 32, 64), (1, 1, 1, 64), (1, 1, 1, 64)],
        [(1, 1, 8, 64), (1, 1, 1, 64), (1, 1, 1, 64)],
    ],
)
@pytest.mark.parametrize("dtypes", [[torch.bfloat16] * 3])
def test_rope_decode_sharding(
    shapes: List[Shape],
    dtypes: List[torch.dtype],
    request,
    device,
):
    def rope_decode_test(*inputs):
        builder: TTNNBuilder = inputs[-1]
        x, sin, cos = inputs[:3]

        result = builder.rotary_embedding_decode(
            x,
            sin=sin,
            cos=cos,
            unit_attrs=["l1_height_sharded"],
        )
        return result

    # Execute the test with sharding enabled
    output_file_mlir = compile_and_execute_ttnn(
        rope_decode_test,
        shapes,
        dtypes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        device=device,
        system_desc_path=request.config.getoption("--sys-desc"),
        target="ttnn",
    )
    print("Output MLIR file:", output_file_mlir)
