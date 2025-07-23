// REQUIRES: stablehlo
// This file incorporates work covered by the following copyright and permission notice:
// SPDX-FileCopyrightText: Copyright (c) 2024 The Shardy Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --stablehlo-pipeline %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir

sdy.mesh @mesh = <["model"=1, "batch"=2]>

func.func @reshape_scalar(%arg0: tensor<1x1xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"model"}, {}]>}) -> tensor<f32> {
  %0 = stablehlo.reshape %arg0 : (tensor<1x1xf32>) -> tensor<f32>
  return %0 : tensor<f32>
}

// CHECK: sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"model"}, {}]>] out_shardings=[<@mesh, []>]
// CHECK: %1 = stablehlo.reshape %arg1 : (tensor<1x1xf32>) -> tensor<f32>
// CHECK: sdy.return %1 : tensor<f32>

func.func @reshape_merge_dim(%arg0: tensor<2x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {"model"}]>}) -> tensor<8xf32> {
  %0 = stablehlo.reshape %arg0 : (tensor<2x4xf32>) -> tensor<8xf32>
  return %0 : tensor<8xf32>
}

// CHECK: sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"batch"}, {"model"}]>] out_shardings=[<@mesh, [{"batch", "model"}]>]
// CHECK: %1 = stablehlo.reshape %arg1 : (tensor<1x4xf32>) -> tensor<4xf32>
// CHECK: sdy.return %1 : tensor<4xf32>

func.func @reshape_split_dim(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}]>}) -> tensor<2x4xf32> {
  %0 = stablehlo.reshape %arg0 : (tensor<8xf32>) -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}

// CHECK: sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"batch"}]>] out_shardings=[<@mesh, [{"batch"}, {}]>]
// CHECK: %1 = stablehlo.reshape %arg1 : (tensor<4xf32>) -> tensor<1x4xf32>
// CHECK: sdy.return %1 : tensor<1x4xf32>

func.func @reshape_split_dim_three_way(%arg0: tensor<4x12xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"batch"}]>}) -> tensor<4x2x3x2xf32> {
  %0 = stablehlo.reshape %arg0 : (tensor<4x12xf32>) -> tensor<4x2x3x2xf32>
  return %0 : tensor<4x2x3x2xf32>
}

// CHECK: sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}, {"batch"}]>] out_shardings=[<@mesh, [{}, {"batch"}, {}, {}]>]
// CHECK: %1 = stablehlo.reshape %arg1 : (tensor<4x6xf32>) -> tensor<4x1x3x2xf32>
// CHECK: sdy.return %1 : tensor<4x1x3x2xf32>

func.func @reshape_split_and_merge_dims(%arg0: tensor<8x4x5xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {"model"}, {}]>}) -> tensor<2x16x5xf32> {
  %0 = stablehlo.reshape %arg0 : (tensor<8x4x5xf32>) -> tensor<2x16x5xf32>
  return %0 : tensor<2x16x5xf32>
}

// CHECK: sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"batch"}, {"model"}, {}]>] out_shardings=[<@mesh, [{"batch"}, {}, {}]>]
// CHECK: %2 = stablehlo.reshape %1 : (tensor<4x4x5xf32>) -> tensor<1x16x5xf32>
// CHECK: sdy.return %2 : tensor<1x16x5xf32>
