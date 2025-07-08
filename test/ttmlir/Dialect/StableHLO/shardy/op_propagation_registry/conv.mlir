// REQUIRES: stablehlo
// This file incorporates work covered by the following copyright and permission notice:
// SPDX-FileCopyrightText: Copyright (c) 2024 The Shardy Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --stablehlo-pipeline %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir

sdy.mesh @mesh = <["model"=1, "batch"=2]>

func.func @conv_simple(%arg0 : tensor<2x224x224x192xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}, {}, {}]>}, %arg1 : tensor<3x3x192x64xf32>) -> tensor<2x112x112x64xf32> {
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {stride = [2, 2], pad = [[0, 1], [0, 1]]} {
      batch_group_count = 1 : i64,
      feature_group_count = 1 : i64,
      lhs_dilations = dense<1> : tensor<2xi64>,
      rhs_dilations = dense<1> : tensor<2xi64>
    } : (tensor<2x224x224x192xf32>, tensor<3x3x192x64xf32>) -> tensor<2x112x112x64xf32>
  return %0 : tensor<2x112x112x64xf32>
}

// CHECK: sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh, [{"batch"}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>] out_shardings=[<@mesh, [{"batch"}, {}, {}, {}]>]
// CHECK: sdy.return %1 : tensor<1x112x112x64xf32>


func.func @conv_feature_group_count(%arg0: tensor<8x224x224x192xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}, {}, {}]>}, %arg1: tensor<3x3x12x256xf32>) -> tensor<8x112x112x256xf32> {
  %0 = stablehlo.convolution(%arg0, %arg1)
    dim_numbers = [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f],
    window = {stride = [2, 2], pad = [[0, 1], [0, 1]]} {
      batch_group_count = 1 : i64,
      feature_group_count = 16 : i64,
      lhs_dilations = dense<1> : tensor<2xi64>,
      rhs_dilations = dense<1> : tensor<2xi64>
    } : (tensor<8x224x224x192xf32>, tensor<3x3x12x256xf32>) -> tensor<8x112x112x256xf32>
  return %0 : tensor<8x112x112x256xf32>
}

// CHECK: sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh, [{"batch"}, {}, {}, {}]>, <@mesh, [{}, {}, {}, {}]>] out_shardings=[<@mesh, [{"batch"}, {}, {}, {}]>]
// CHECK: sdy.return %1 : tensor<4x112x112x256xf32>
