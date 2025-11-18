# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import List, Optional
from builder.base.builder import Operand, Shape
from builder.ttnn.ttnn_builder import TTNNBuilder
from builder.base.builder_utils import compile_and_execute_ttnn

pytestmark = pytest.mark.frontend("ttir")


def check_op(mlir_file: str, op_name: str) -> bool:
    """Check if an operation exists in the MLIR file."""
    with open(mlir_file, "r") as f:
        for line in f:
            if f"ttnn.{op_name}" in line:
                return True
    return False


@pytest.mark.parametrize(
    "shapes",
    [
        [
            (64, 128),
            (128, 256),
        ]
    ],
)
@pytest.mark.parametrize(
    "activation",
    [("sigmoid", torch.sigmoid, lambda b, x, attrs: b.sigmoid(x, unit_attrs=attrs))],
)
@pytest.mark.parametrize("dtypes", [[torch.float32] * 2])
def test_matmul_activation_fusing(
    shapes: List[Shape],
    activation: tuple,
    dtypes: List[torch.dtype],
    request,
    device,
):
    activation_name, torch_activation_fn, builder_activation_fn = activation

    def matmul_sigmoid(
        input_tensor: Operand,
        weight: Operand,
        builder: TTNNBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        input_tensor_data = torch.randn(shapes[0], dtype=dtypes[0])
        weight_data = torch.randn(shapes[1], dtype=dtypes[1])

        matmul_result = torch.matmul(input_tensor_data, weight_data)
        golden_output = torch_activation_fn(matmul_result)

        matmul = builder.matmul(input_tensor, weight, unit_attrs=unit_attrs)
        activation_op = builder_activation_fn(builder, matmul, unit_attrs)

        builder.set_goldens(
            {
                input_tensor: input_tensor_data,
                weight: weight_data,
            },
            {activation_op: golden_output},
        )
        return activation_op

    output = compile_and_execute_ttnn(
        matmul_sigmoid,
        shapes,
        dtypes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        device=device,
    )

    assert check_op(output, "matmul"), "Matmul operation should exist"
    assert not check_op(
        output, activation_name
    ), f"Standalone {activation_name} operation should be fused"


@pytest.mark.xfail(
    reason="Fails golden, see https://github.com/tenstorrent/tt-mlir/issues/5789"
)
@pytest.mark.parametrize(
    "shapes",
    [
        [(64, 128), (256, 128), (256,)],
    ],
)
@pytest.mark.parametrize(
    "activation",
    [("sigmoid", torch.sigmoid, lambda b, x, attrs: b.sigmoid(x, unit_attrs=attrs))],
)
@pytest.mark.parametrize("dtypes", [[torch.float32] * 3])
def test_linear_activation_fusing(
    shapes: List[Shape],
    activation: tuple,
    dtypes: List[torch.dtype],
    request,
    device,
):
    activation_name, torch_activation_fn, builder_activation_fn = activation

    def linear_sigmoid(
        input_tensor: Operand,
        weight: Operand,
        bias: Operand,
        builder: TTNNBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        input_tensor_data = torch.randn(shapes[0], dtype=dtypes[0])
        weight_data = torch.randn(shapes[1], dtype=dtypes[1])
        bias_data = torch.randn(shapes[2], dtype=dtypes[2])

        goldens = {
            input_tensor: input_tensor_data,
            weight: weight_data,
            bias: bias_data,
        }

        linear_result = torch.nn.functional.linear(
            input_tensor_data, weight_data, bias=bias_data
        )

        golden_output = torch_activation_fn(linear_result)

        linear = builder.linear(
            input_tensor, weight, bias=bias, transpose_b=True, unit_attrs=unit_attrs
        )
        activation_op = builder_activation_fn(builder, linear, unit_attrs)

        builder.set_goldens(goldens, {activation_op: golden_output})
        return activation_op

    output = compile_and_execute_ttnn(
        linear_sigmoid,
        shapes,
        dtypes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        device=device,
    )

    assert check_op(output, "linear"), "Linear operation should exist"
    assert not check_op(
        output, activation_name
    ), f"Standalone {activation_name} operation should be fused"
