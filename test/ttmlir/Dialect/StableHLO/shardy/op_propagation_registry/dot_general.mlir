// REQUIRES: stablehlo
// This file incorporates work covered by the following copyright and permission notice:
// SPDX-FileCopyrightText: Copyright (c) 2024 The Shardy Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --stablehlo-pipeline %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir

sdy.mesh @mesh = <["model"=1, "batch"=2]>

func.func @dot_general_batching_dims(%arg0: tensor<4x8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}, {}]>}, %arg1: tensor<4x32x16xf32>) -> tensor<4x8x16xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<4x8x32xf32>, tensor<4x32x16xf32>) -> tensor<4x8x16xf32>
  return %0 : tensor<4x8x16xf32>
}

// CHECK: sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh, [{"batch"}, {}, {}]>, <@mesh, [{"batch"}, {}, {}]>] out_shardings=[<@mesh, [{"batch"}, {}, {}]>]
// CHECK: stablehlo.dot_general %arg2, %arg3, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<2x8x32xf32>, tensor<2x32x16xf32>) -> tensor<2x8x16xf32>
// CHECK: sdy.return %1 : tensor<2x8x16xf32>
