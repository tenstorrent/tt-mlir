// REQUIRES: stablehlo
// This file incorporates work covered by the following copyright and permission notice:
// SPDX-FileCopyrightText: Copyright (c) 2024 The Shardy Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --stablehlo-pipeline -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

sdy.mesh @mesh = <["model"=1, "batch"=2]>

func.func @broadcast_in_dim(%arg0: tensor<2x13x1xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}, {}]>}) -> tensor<2x64x13x1xf32> {
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 2, 3] : (tensor<2x13x1xf32>) -> tensor<2x64x13x1xf32>
  return %0 :  tensor<2x64x13x1xf32>
}

// CHECK: sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"batch"}, {}, {}]>] out_shardings=[<@mesh, [{"batch"}, {}, {}, {}]>]
// CHECK: %1 = stablehlo.broadcast_in_dim %arg1, dims = [0, 2, 3] : (tensor<1x13x1xf32>) -> tensor<1x64x13x1xf32>
// CHECK: sdy.return %1 : tensor<1x64x13x1xf32>

func.func @broadcast_in_dim_size_zero_dim(%arg0: tensor<2x13x0xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}, {}]>}) -> tensor<2x64x13x0xf32> {
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 2, 3] : (tensor<2x13x0xf32>) -> tensor<2x64x13x0xf32>
  return %0 :  tensor<2x64x13x0xf32>
}

// CHECK: sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"batch"}, {}, {}]>] out_shardings=[<@mesh, [{"batch"}, {}, {}, {}]>]
// CHECK: %1 = stablehlo.broadcast_in_dim %arg1, dims = [0, 2, 3] : (tensor<1x13x0xf32>) -> tensor<1x64x13x0xf32>
// CHECK: sdy.return %1 : tensor<1x64x13x0xf32>
