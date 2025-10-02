# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import List, Optional
from builder.base.builder import Operand, Shape
from builder.base.builder_golden import BuilderGoldenTensor
from builder.ttir.ttir_builder import TTIRBuilder
from builder.base.builder_utils import compile_ttir_to_flatbuffer

pytestmark = pytest.mark.frontend("ttir")


def check_op(mlir_file: str, op_name: str) -> bool:
    with open(mlir_file, "r") as f:
        for line in f:
            if f"ttnn.{op_name}" in line:
                return True
    return False


@pytest.mark.parametrize(
    "shapes",
    [
        [
            (1, 32, 32, 64),
            (64, 32, 3, 3),
            (1, 1, 1, 64),
            (16,),  # batch_norm scale (gamma)
            (16,),  # batch_norm offset (beta)
            (16,),  # batch_norm mean
            (16,),  # batch_norm variance
        ]
    ],
)
@pytest.mark.parametrize("dtypes", [[torch.float32] * 7])
@pytest.mark.parametrize(
    "stride,padding,dilation,groups", [([2, 1], [2, 1], [2, 1], 2)]
)
@pytest.mark.parametrize("dimension", [1])  # channel dimension for NCHW format
@pytest.mark.parametrize("epsilon", [0.0])
@pytest.mark.parametrize("training", [False])
def test_batch_norm_decomposition(
    shapes: List[Shape],
    dtypes: List[torch.dtype],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    groups: int,
    dimension: int,
    epsilon: float,
    training: bool,
    request,
):
    def conv2d_batch_norm(
        input_tensor: Operand,
        conv_weight: Operand,
        conv_bias: Operand,
        bn_scale: Operand,
        bn_offset: Operand,
        bn_mean: Operand,
        bn_variance: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        # Create input tensor with random data
        input_tensor_data = torch.randn(shapes[0], dtype=dtypes[0])

        # Create conv2d weights and bias
        conv_weight_data = torch.randn(shapes[1], dtype=dtypes[1])
        conv_bias_data = torch.randn(shapes[2], dtype=dtypes[2])

        # Create batch norm parameters
        bn_scale_data = torch.randn(shapes[3], dtype=dtypes[3])
        bn_offset_data = torch.randn(shapes[4], dtype=dtypes[4])
        bn_mean_data = torch.randn(shapes[5], dtype=dtypes[5])
        bn_variance_data = (
            torch.abs(torch.randn(shapes[6], dtype=dtypes[6])) + 1e-5
        )  # Ensure positive variance

        input_tensor_data_rs = input_tensor_data.transpose(-2, -1).transpose(-3, -2)
        conv_result = torch.nn.functional.conv2d(
            input_tensor_data_rs,
            conv_weight_data,
            conv_bias_data.squeeze(),
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        conv_result = conv_result.transpose(-3, -2).transpose(-2, -1)

        golden_output = torch.nn.functional.batch_norm(
            conv_result,
            bn_mean_data,
            bn_variance_data,
            bn_scale_data,
            bn_offset_data,
            training=training,
            eps=epsilon,
        )

        conv2d_0 = builder.conv2d(
            input_tensor,
            conv_weight,
            conv_bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            unit_attrs=unit_attrs,
        )

        batch_norm_0 = builder.batch_norm(
            conv2d_0,
            bn_scale,
            bn_offset,
            bn_mean,
            bn_variance,
            epsilon=epsilon,
            dimension=dimension,
            training=training,
            unit_attrs=unit_attrs,
        )

        builder.set_goldens(
            {
                input_tensor: input_tensor_data,
                conv_weight: conv_weight_data,
                conv_bias: conv_bias_data,
                bn_scale: bn_scale_data,
                bn_offset: bn_offset_data,
                bn_mean: bn_mean_data,
                bn_variance: bn_variance_data,
            },
            {batch_norm_0: golden_output},
        )
        builder.set_operand_goldens({conv2d_0: conv_result})
        return batch_norm_0

    output = compile_ttir_to_flatbuffer(
        conv2d_batch_norm,
        shapes,
        dtypes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
        pipeline_options=["enable-fusing-conv2d-with-multiply-pattern=true"],
    )
    assert check_op(output, "conv2d") and not check_op(output, "batch_norm")


@pytest.mark.parametrize(
    "shapes",
    [
        [
            (1, 32, 32, 64),  # input
            (64, 64, 3, 3),  # conv weight
            (1, 1, 1, 64),  # conv bias
        ]
    ],
)
@pytest.mark.parametrize("dtypes", [[torch.float32] * 3])
@pytest.mark.parametrize("stride", [[1, 1]])
@pytest.mark.parametrize("padding", [[1, 1]])
@pytest.mark.parametrize("dilation", [[1, 1]])
@pytest.mark.parametrize("groups", [1])
@pytest.mark.parametrize("activation", ["relu", "relu6", "silu"])
def test_conv_activation_fusing(
    shapes: List[Shape],
    dtypes: List[torch.dtype],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    groups: int,
    activation: str,
    request,
):
    def conv2d_activation(
        input_tensor: Operand,
        conv_weight: Operand,
        conv_bias: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        # Create input tensor with random data
        input_tensor_data = torch.randn(shapes[0], dtype=dtypes[0])

        # Create conv2d weights and bias
        conv_weight_data = torch.randn(shapes[1], dtype=dtypes[1])
        conv_bias_data = torch.randn(shapes[2], dtype=dtypes[2])

        # Calculate golden output using torch operations
        input_tensor_data_rs = input_tensor_data.transpose(-2, -1).transpose(-3, -2)
        conv_result = torch.nn.functional.conv2d(
            input_tensor_data_rs,
            conv_weight_data,
            conv_bias_data.squeeze(),
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        conv_result = conv_result.transpose(-3, -2).transpose(-2, -1)

        # Apply activation based on parameter
        if activation == "relu":
            golden_output = torch.nn.functional.relu(conv_result)
        elif activation == "relu6":
            golden_output = torch.nn.functional.relu6(conv_result)
        elif activation == "silu":
            golden_output = torch.nn.functional.silu(conv_result)

        # Create conv2d builder op
        conv = builder.conv2d(
            input_tensor,
            conv_weight,
            conv_bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            unit_attrs=unit_attrs,
        )

        # Add activation builder op based on parameter
        if activation == "relu":
            activation_op = builder.relu(conv)
        elif activation == "relu6":
            activation_op = builder.relu6(conv)
        elif activation == "silu":
            activation_op = builder.silu(conv)

        builder.set_goldens(
            {
                input_tensor: input_tensor_data,
                conv_weight: conv_weight_data,
                conv_bias: conv_bias_data,
            },
            {conv: golden_output},
        )
        return activation_op

    output = compile_ttir_to_flatbuffer(
        conv2d_activation,
        shapes,
        dtypes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )
    assert check_op(output, "conv2d") and not check_op(output, activation)


@pytest.mark.parametrize(
    "shapes",
    [
        [
            (1, 32, 32, 64),  # input
            (64, 64, 3, 3),  # conv weight
            (1, 1, 1, 64),  # conv bias
        ]
    ],
)
@pytest.mark.parametrize("dtypes", [[torch.float32] * 3])
@pytest.mark.parametrize("stride", [[1, 1]])
@pytest.mark.parametrize("padding", [[1, 1]])
@pytest.mark.parametrize("dilation", [[1, 1]])
@pytest.mark.parametrize("groups", [1])

# Test fusing when silu is decomposed as x * sigmoid(x)
def test_conv_silu_decomposed_fusing(
    shapes: List[Shape],
    dtypes: List[torch.dtype],
    stride: List[int],
    padding: List[int],
    dilation: List[int],
    groups: int,
    request,
):
    def conv2d_silu_decomposed(
        input_tensor: Operand,
        conv_weight: Operand,
        conv_bias: Operand,
        builder: TTIRBuilder,
        unit_attrs: Optional[List[str]] = None,
    ):
        # Create input tensor with random data
        input_tensor_data = torch.randn(shapes[0], dtype=dtypes[0])

        # Create conv2d weights and bias
        conv_weight_data = torch.randn(shapes[1], dtype=dtypes[1])
        conv_bias_data = torch.randn(shapes[2], dtype=dtypes[2])

        # Calculate golden output using torch operations
        input_tensor_data_rs = input_tensor_data.transpose(-2, -1).transpose(-3, -2)
        conv_result = torch.nn.functional.conv2d(
            input_tensor_data_rs,
            conv_weight_data,
            conv_bias_data.squeeze(),
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        conv_result = conv_result.transpose(-3, -2).transpose(-2, -1)
        golden_output = conv_result * torch.sigmoid(conv_result)

        # Create conv2d builder op
        conv = builder.conv2d(
            input_tensor,
            conv_weight,
            conv_bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            unit_attrs=unit_attrs,
        )

        # Add builder ops for x * sigmoid(x)
        sigmoid_op = builder.sigmoid(conv, unit_attrs=unit_attrs)
        silu_decomposed = builder.multiply(conv, sigmoid_op, unit_attrs=unit_attrs)

        builder.set_goldens(
            {
                input_tensor: input_tensor_data,
                conv_weight: conv_weight_data,
                conv_bias: conv_bias_data,
            },
            {conv: golden_output},
        )
        return silu_decomposed

    output = compile_ttir_to_flatbuffer(
        conv2d_silu_decomposed,
        shapes,
        dtypes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc"),
    )
    assert (
        check_op(output, "conv2d")
        and not check_op(output, "sigmoid")
        and not check_op(output, "multiply")
    )
