# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional

import pytest
import torch

from builder.base.builder import Operand, Shape
from builder.base.builder_utils import compile_and_execute_ttnn
from builder.ttnn.ttnn_builder import TTNNBuilder

pytestmark = pytest.mark.frontend("ttnn")


@pytest.mark.parametrize(
    "shapes",
    [
        [(1, 1, 32, 2048), (2048,)],
    ],
)
@pytest.mark.parametrize("dtypes", [[torch.bfloat16] * 2])
@pytest.mark.parametrize("epsilon", [1.0e-5])
def test_rmsnorm_sharding(
    shapes: List[Shape],
    dtypes: List[torch.dtype],
    epsilon: float,
    request,
    device,
):
    def rmsnorm_test(
        input_tensor: Operand,
        weight: Operand,
        builder: TTNNBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):

        result = builder.rms_norm(
            input_tensor,
            weight=weight,
            epsilon=epsilon,
            unit_attrs=["l1_width_sharded"],
        )
        return result

    # Execute the test with sharding enabled
    output_file_mlir = compile_and_execute_ttnn(
        rmsnorm_test,
        shapes,
        dtypes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        device=device,
        system_desc_path=request.config.getoption("--sys-desc"),
        target="ttnn",
    )
    print("Output MLIR file:", output_file_mlir)
