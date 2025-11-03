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


def check_op_with_activation(mlir_file: str, op_name: str, activation: str) -> bool:
    """Check if an operation has a specific activation attribute in the MLIR file."""
    with open(mlir_file, "r") as f:
        content = f.read()
        # Look for patterns like: ttnn.conv2d ... activation = <UnaryOpType.relu>
        # or: ttnn.matmul ... activation = "sigmoid"
        op_pattern = f"ttnn.{op_name}"
        for line in content.splitlines():
            if op_pattern in line:
                # Check if activation is in this line or nearby lines
                # Search in a window around this line
                lines = content.splitlines()
                idx = lines.index(line)
                # Check current and next few lines for activation
                for i in range(idx, min(idx + 5, len(lines))):
                    if f"activation" in lines[i] and activation in lines[i]:
                        return True
    return False


@pytest.mark.parametrize(
    "shapes",
    [
        [
            (64, 128),  # input
            (128, 256),  # weight
        ]
    ],
)
@pytest.mark.parametrize("dtypes", [[torch.float32] * 2])
def test_matmul_sigmoid_fusing(
    shapes: List[Shape],
    dtypes: List[torch.dtype],
    request,
    device,
):
    """Test that Matmul fuses with Sigmoid activation."""

    def matmul_sigmoid(
        input_tensor: Operand,
        weight: Operand,
        builder: TTNNBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        # Create input tensors with random data
        input_tensor_data = torch.randn(shapes[0], dtype=dtypes[0])
        weight_data = torch.randn(shapes[1], dtype=dtypes[1])

        # Calculate golden output
        matmul_result = torch.matmul(input_tensor_data, weight_data)
        golden_output = torch.sigmoid(matmul_result)

        # Create matmul and sigmoid ops
        matmul = builder.matmul(input_tensor, weight, unit_attrs=unit_attrs)
        sigmoid_op = builder.sigmoid(matmul, unit_attrs=unit_attrs)

        builder.set_goldens(
            {
                input_tensor: input_tensor_data,
                weight: weight_data,
            },
            {sigmoid_op: golden_output},
        )
        return sigmoid_op

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
        output, "sigmoid"
    ), "Standalone sigmoid operation should be fused"


@pytest.mark.parametrize(
    "shapes",
    [
        [
            (64, 128),  # input
            (256, 128),  # weight (note: shape for transpose_b=true)
        ]
    ],
)
@pytest.mark.parametrize("dtypes", [[torch.float32] * 2])
def test_linear_sigmoid_fusing(
    shapes: List[Shape],
    dtypes: List[torch.dtype],
    request,
    device,
):
    """Test that Linear fuses with Sigmoid activation."""

    def linear_sigmoid(
        input_tensor: Operand,
        weight: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        # Create input tensors with random data
        input_tensor_data = torch.randn(shapes[0], dtype=dtypes[0])
        weight_data = torch.randn(shapes[1], dtype=dtypes[1])

        # Calculate golden output - linear does y = x @ W^T
        linear_result = torch.nn.functional.linear(
            input_tensor_data, weight_data, bias=None
        )
        golden_output = torch.sigmoid(linear_result)

        # Create linear and sigmoid ops
        linear = builder.linear(
            input_tensor, weight, transpose_b=True, unit_attrs=unit_attrs
        )
        sigmoid_op = builder.sigmoid(linear, unit_attrs=unit_attrs)

        builder.set_goldens(
            {
                input_tensor: input_tensor_data,
                weight: weight_data,
            },
            {sigmoid_op: golden_output},
        )
        return sigmoid_op

    output = compile_and_execute_ttir(
        linear_sigmoid,
        shapes,
        dtypes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        device=device,
    )

    # Verify that:
    # 1. Linear operation exists
    # 2. Sigmoid operation was removed (fused into linear)
    assert check_op(output, "linear"), "Linear operation should exist"
    assert not check_op(
        output, "sigmoid"
    ), "Standalone sigmoid operation should be fused"


@pytest.mark.parametrize(
    "shapes",
    [
        [
            (64, 128),  # input
            (256, 128),  # weight (note: shape for transpose_b=true)
            (256,),  # bias
        ]
    ],
)
@pytest.mark.parametrize("dtypes", [[torch.float32] * 3])
def test_linear_bias_sigmoid_fusing(
    shapes: List[Shape],
    dtypes: List[torch.dtype],
    request,
    device,
):
    """Test that Linear with bias fuses with Sigmoid activation."""

    def linear_bias_sigmoid(
        input_tensor: Operand,
        weight: Operand,
        bias: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        # Create input tensors with random data
        input_tensor_data = torch.randn(shapes[0], dtype=dtypes[0])
        weight_data = torch.randn(shapes[1], dtype=dtypes[1])
        bias_data = torch.randn(shapes[2], dtype=dtypes[2])

        # Calculate golden output - linear does y = x @ W^T + b
        linear_result = torch.nn.functional.linear(
            input_tensor_data, weight_data, bias=bias_data
        )
        golden_output = torch.sigmoid(linear_result)

        # Create linear and sigmoid ops
        linear = builder.linear(
            input_tensor, weight, bias=bias, transpose_b=True, unit_attrs=unit_attrs
        )
        sigmoid_op = builder.sigmoid(linear, unit_attrs=unit_attrs)

        builder.set_goldens(
            {
                input_tensor: input_tensor_data,
                weight: weight_data,
                bias: bias_data,
            },
            {sigmoid_op: golden_output},
        )
        return sigmoid_op

    output = compile_and_execute_ttnn(
        linear_bias_sigmoid,
        shapes,
        dtypes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        device=device,
    )

    # Verify that:
    # 1. Linear operation exists
    # 2. Sigmoid operation was removed (fused into linear)
    assert check_op(output, "linear"), "Linear operation should exist"
    assert not check_op(
        output, "sigmoid"
    ), "Standalone sigmoid operation should be fused"
