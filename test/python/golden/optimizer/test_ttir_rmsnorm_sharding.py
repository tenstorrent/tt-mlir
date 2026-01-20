# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import re
from typing import List, Optional

import pytest
import torch

from builder.base.builder_utils import Operand, Shape
from builder.base.builder_apis import compile_and_execute_ttir
from builder.ttir.ttir_builder import TTIRBuilder
from conftest import get_request_kwargs

# Temporarily disabled: RMSNorm with sharded input causes crash in metal.
pytestmark = [
    pytest.mark.frontend("ttir"),
    pytest.mark.skip(
        reason="Temporarily disabled: RMSNorm sharding causes crash in metal"
    ),
]


def check_sharded_output(mlir_file: str, op_name: str):
    sharded_layouts = []
    with open(mlir_file, "r") as f:
        for line in f:
            if line.startswith("#ttnn_layout") and "sharded" in line:
                layout = line.split("=", 1)[0].strip()
                sharded_layouts.append(layout)

            if len(sharded_layouts) > 0:
                pattern = re.compile(
                    rf".*{op_name}.*->.*({'|'.join(sharded_layouts)}).*"
                )
                if pattern.search(line):
                    return True
    return False


@pytest.mark.skip(
    "Causes segfault during pipeline, see https://github.com/tenstorrent/tt-mlir/issues/5283"
)
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
    def module(builder: TTIRBuilder):
        @builder.func(shapes, dtypes)
        def rmsnorm_test(
            input_tensor: Operand,
            weight: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            # Create torch tensors for golden reference
            torch.manual_seed(42)
            input_shape = shapes[0]
            weight_shape = shapes[1]
            torch_input = torch.randn(input_shape, dtype=dtypes[0])
            torch_weight = torch.randn(weight_shape, dtype=dtypes[1])

            result = builder.rms_norm(
                input_tensor,
                normalized_shape=weight_shape,
                weight=weight,
                epsilon=epsilon,
                unit_attrs=unit_attrs,
            )

            builder.set_goldens(
                {input_tensor: torch_input, weight: torch_weight},
                {
                    result: torch.rms_norm(
                        torch_input,
                        normalized_shape=weight_shape,
                        weight=torch_weight,
                        eps=epsilon,
                    )
                },
            )
            return result

    output_file_mlir = compile_and_execute_ttir(
        module,
        **get_request_kwargs(request),
        device=device,
        pipeline_options=[
            "enable-optimizer=true",
            "memory-layout-analysis-enabled=true",
        ],
        target="ttnn",
    )

    assert check_sharded_output(
        output_file_mlir, "rms_norm"
    ), "RMSNorm operation should have sharded output layouts"
