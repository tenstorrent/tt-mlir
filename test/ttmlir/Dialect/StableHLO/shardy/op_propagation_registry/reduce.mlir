// REQUIRES: stablehlo
// This file incorporates work covered by the following copyright and permission notice:
// SPDX-FileCopyrightText: Copyright (c) 2024 The Shardy Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --stablehlo-pipeline -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

sdy.mesh @mesh = <["model"=1, "batch"=2]>

func.func @reduce_single_result(%arg0: tensor<2x64x13xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}, {}]>}) -> tensor<2x13xf32> {
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.reduce(%arg0 init: %0) applies stablehlo.add across dimensions = [1] : (tensor<2x64x13xf32>, tensor<f32>) -> tensor<2x13xf32>
  return %1 : tensor<2x13xf32>
}

// CHECK: sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"batch"}, {}, {}]>] out_shardings=[<@mesh, [{"batch"}, {}]>]
// CHECK: %1 = stablehlo.reduce(%arg1 init: %cst) applies stablehlo.add across dimensions = [1] : (tensor<1x64x13xf32>, tensor<f32>) -> tensor<1x13xf32>
// CHECK: sdy.return %1 : tensor<1x13xf32>
