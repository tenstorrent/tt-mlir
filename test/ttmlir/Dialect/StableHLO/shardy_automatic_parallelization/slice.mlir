// REQUIRES: stablehlo
// This file incorporates work covered by the following copyright and permission notice:
// SPDX-FileCopyrightText: Copyright (c) 2024 The Shardy Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --automatic-sharding-pipeline="mesh-shape=1,2" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir

sdy.mesh @mesh = <["model"=1, "batch"=2]>

func.func @concat_operands_are_results_of_slices(%arg0: tensor<4x40x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}, {}]>}, %arg1: tensor<4x60x256xf32>) -> tensor<4x80x256xf32> {
  %0 = stablehlo.slice %arg0 [0:4, 0:32, 0:256] : (tensor<4x40x256xf32>) -> tensor<4x32x256xf32>
  %1 = stablehlo.slice %arg1 [0:4, 0:48, 0:256] : (tensor<4x60x256xf32>) -> tensor<4x48x256xf32>
  %2 = stablehlo.concatenate %0, %1, dim = 1 : (tensor<4x32x256xf32>, tensor<4x48x256xf32>) -> tensor<4x80x256xf32>
  return %2 : tensor<4x80x256xf32>
}

// CHECK: sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh, [{"batch"}, {}, {}]>, <@mesh, [{"batch"}, {}, {}]>] out_shardings=[<@mesh, [{"batch"}, {}, {}]>]
// CHECK: stablehlo.slice %arg2 [0:2, 0:32, 0:256] : (tensor<2x40x256xf32>) -> tensor<2x32x256xf32>
// CHECK: stablehlo.slice %arg3 [0:2, 0:48, 0:256] : (tensor<2x60x256xf32>) -> tensor<2x48x256xf32>
// CHECK: sdy.return %3 : tensor<2x80x256xf32>

func.func @concat_operands_are_from_slices_of_the_same_tensor(%arg0: tensor<4x40x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}, {}]>}) -> tensor<4x96x256xf32> {
  %0 = stablehlo.slice %arg0 [0:4, 0:32, 0:256] : (tensor<4x40x256xf32>) -> tensor<4x32x256xf32>
  %1 = stablehlo.slice %arg0 [0:4, 0:24, 0:256] : (tensor<4x40x256xf32>) -> tensor<4x24x256xf32>
  %2 = stablehlo.concatenate %0, %arg0, %1, dim = 1 : (tensor<4x32x256xf32>, tensor<4x40x256xf32>, tensor<4x24x256xf32>) -> tensor<4x96x256xf32>
  return %2 : tensor<4x96x256xf32>
}

// CHECK: sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"batch"}, {}, {}]>] out_shardings=[<@mesh, [{"batch"}, {}, {}]>]
// CHECK: stablehlo.slice %arg1 [0:2, 0:32, 0:256] : (tensor<2x40x256xf32>) -> tensor<2x32x256xf32>
// CHECK: stablehlo.slice %arg1 [0:2, 0:24, 0:256] : (tensor<2x40x256xf32>) -> tensor<2x24x256xf32>
// CHECK: sdy.return %3 : tensor<2x96x256xf32>

func.func @non_batch_dim_slice(%arg0: tensor<4x40x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"batch"}, {}]>}, %arg1: tensor<4x60x256xf32>) -> tensor<4x80x256xf32> {
  %0 = stablehlo.slice %arg0 [0:4, 0:32, 0:256] : (tensor<4x40x256xf32>) -> tensor<4x32x256xf32>
  %1 = stablehlo.slice %arg1 [0:4, 0:48, 0:256] : (tensor<4x60x256xf32>) -> tensor<4x48x256xf32>
  %2 = stablehlo.concatenate %0, %1, dim = 1 : (tensor<4x32x256xf32>, tensor<4x48x256xf32>) -> tensor<4x80x256xf32>
  return %2 : tensor<4x80x256xf32>
}

// CHECK: sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh, [{}, {"batch"}, {}]>, <@mesh, [{}, {"batch"}, {}]>] out_shardings=[<@mesh, [{}, {"batch"}, {}]>]
// CHECK: stablehlo.slice %arg2 [0:4, 0:16, 0:256] : (tensor<4x20x256xf32>) -> tensor<4x16x256xf32>
// CHECK: stablehlo.slice %arg3 [0:4, 0:24, 0:256] : (tensor<4x30x256xf32>) -> tensor<4x24x256xf32>
// CHECK: sdy.return %3 : tensor<4x40x256xf32>
