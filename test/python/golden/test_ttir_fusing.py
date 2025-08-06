# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import List, Optional
from builder.base.builder import Operand, Shape
from builder.ttir.ttir_builder import TTIRBuilder
from builder.ttir.ttir_utils import compile_ttir_to_flatbuffer

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
            (1, 16, 32, 64),
            (16,),              # batch_norm scale (gamma)
            (16,),              # batch_norm offset (beta)
            (16,),              # batch_norm mean
            (16,),              # batch_norm variance
        ]
    ],
)
@pytest.mark.parametrize("dtypes", [[torch.float32] * 8])
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
        conv_output: Operand,
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
        bn_scale_data = torch.randn(shapes[4], dtype=dtypes[4])
        bn_offset_data = torch.randn(shapes[5], dtype=dtypes[5])
        bn_mean_data = torch.randn(shapes[6], dtype=dtypes[6])
        bn_variance_data = torch.abs(torch.randn(shapes[7], dtype=dtypes[7])) + 1e-5  # Ensure positive variance

        input_tensor_data_rs = input_tensor_data.transpose(-2, -1).transpose(-3, -2)
        conv_result = torch.nn.functional.conv2d(
            input_tensor_data_rs, conv_weight_data, conv_bias_data.squeeze(), stride=stride, padding=padding, dilation=dilation, groups=groups
        )
        conv_result = conv_result.transpose(-3, -2).transpose(-2, -1)

        golden_output = torch.nn.functional.batch_norm(
            conv_result, bn_mean_data, bn_variance_data, bn_scale_data, bn_offset_data, training=training, eps=epsilon
        )
        
        builder.set_graph_input_output(
            [input_tensor_data, conv_weight_data, conv_bias_data, conv_result, bn_scale_data, bn_offset_data, bn_mean_data, bn_variance_data], 
            [golden_output], 
            override=True
        )

        conv_result = builder.conv2d(
            input_tensor,
            conv_weight,
            conv_bias,
            conv_output,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            unit_attrs=unit_attrs,
        )

        return builder.batch_norm(
            conv_result,
            bn_scale,
            bn_offset,
            bn_mean,
            bn_variance,
            epsilon=epsilon,
            dimension=dimension,
            training=training,
            unit_attrs=unit_attrs,
        )

    output = compile_ttir_to_flatbuffer(
        conv2d_batch_norm,
        shapes,
        dtypes,
        test_base=request.node.name,
        output_root=request.config.getoption("--path"),
        system_desc_path=request.config.getoption("--sys-desc")
    )
    assert check_op(output, "conv2d") and not check_op(output, "batch_norm")