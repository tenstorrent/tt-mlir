# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from typing import Callable, List, Optional, Tuple, Union
from collections import OrderedDict
from functools import reduce
import operator

from builder.base.builder_utils import build_module, load_mlir_file, split_mlir_file

pytestmark = pytest.mark.frontend("ttir")

resnet_ops_ir = """module {
  func.func @model(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>, %arg2: tensor<32x32xf32>, %arg3: tensor<12x3x224x224xf32>, %arg4: tensor<64x3x7x7xf32>, %arg5: tensor<12x64x112x112xf32>, %arg6: tensor<64xf32>, %arg7: tensor<64xf32>, %arg8: tensor<64xf32>, %arg9: tensor<64xf32>, %arg10: tensor<12x64x114x114xf32>) -> tensor<1x1024xf32> {
    %0 = \"ttir.constant\"() <{value = dense<-1.19942093> : tensor<32x32xf32>}> : () -> tensor<32x32xf32>
    %1 = ttir.empty() : tensor<32x34xf32>
    %2 = \"ttir.pad\"(%arg0, %1) <{padding = array<i32: 0, 0, 1, 1>, value = 0.000000e+00 : f32}> : (tensor<32x32xf32>, tensor<32x34xf32>) -> tensor<32x34xf32>
    %3 = \"ttir.dot_general\"(%arg0, %arg1) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    %4 = ttir.empty() : tensor<32x32xf32>
    %5 = \"ttir.permute\"(%arg0, %4) <{permutation = array<i64: 1, 0>}> : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    %6 = ttir.empty() : tensor<32x32xf32>
    %7 = \"ttir.broadcast\"(%arg0, %6) <{broadcast_dimensions = array<i64: 1, 1>}> : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    %8 = ttir.empty() : tensor<32x32xf32>
    %9 = \"ttir.add\"(%arg0, %arg1, %8) : (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    %10 = ttir.empty() : tensor<32xf32>
    %11 = \"ttir.sum\"(%9, %10) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<32x32xf32>, tensor<32xf32>) -> tensor<32xf32>
    %12 = ttir.empty() : tensor<32x32xf32>
    %13 = \"ttir.multiply\"(%11, %arg2, %12) : (tensor<32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    %14 = ttir.empty() : tensor<32x32xf32>
    %15 = \"ttir.maximum\"(%13, %arg0, %14) : (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    %16 = ttir.empty() : tensor<1x1024xf32>
    %17 = \"ttir.reshape\"(%arg0, %16) <{shape = [1 : i32, 1024 : i32]}> : (tensor<32x32xf32>, tensor<1x1024xf32>) -> tensor<1x1024xf32>
    %18 = ttir.empty() : tensor<12x64x112x112xf32>
    %19 = \"ttir.convolution\"(%arg3, %arg4, %18) <{batch_group_count = 1 : i64, convolution_layout = #ttir<convolution_layout input_batch = 0, input_feature = 1, input_spatial_dimensions = 2x3, kernel_output_feature = 0, kernel_input_feature = 1, kernel_spatial_dimensions = 2x3, output_batch = 0, output_feature = 1, output_spatial_dimensions = 2x3>, feature_group_count = 1 : i64, input_dilation = array<i64: 1, 1>, padding = array<i64: 3, 3, 3, 3>, weight_dilation = array<i64: 1, 1>, window_reversal = array<i1: false, false>, window_strides = array<i64: 2, 2>}> : (tensor<12x3x224x224xf32>, tensor<64x3x7x7xf32>, tensor<12x64x112x112xf32>) -> tensor<12x64x112x112xf32>
    %20 = ttir.empty() : tensor<12x64x112x112xf32>
    %21 = \"ttir.batch_norm_inference\"(%arg5, %arg6, %arg7, %arg8, %arg9, %20) <{dimension = 1 : i32, epsilon = 9.99999974E-6 : f32}> : (tensor<12x64x112x112xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<12x64x112x112xf32>) -> tensor<12x64x112x112xf32>
    %22 = ttir.empty() : tensor<12x31x56x114xf32>
    %23 = \"ttir.pooling\"(%arg10, %22) <{base_dilations = array<i64: 1, 1, 1, 1>, operandSegmentSizes = array<i32: 1, 1>, padding = array<i64: 0, 0, 0, 0, 0, 0, 0, 0>, pooling_method = #ttir<pooling_method Max>, window_dilations = array<i64: 1, 1, 1, 1>, window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>}> : (tensor<12x64x114x114xf32>, tensor<12x31x56x114xf32>) -> tensor<12x31x56x114xf32>
    return %17 : tensor<1x1024xf32>
  }
}"""


@pytest.mark.parametrize("mlir_text", [resnet_ops_ir])
def test_resnet_ops(mlir_text: str, request, device):
    mlir_module, builder = load_mlir_file(mlir_text)
    builder_module_list = split_mlir_file(mlir_module, builder)
