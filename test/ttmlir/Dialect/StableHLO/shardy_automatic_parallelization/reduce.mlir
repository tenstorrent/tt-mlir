// REQUIRES: stablehlo
// This file incorporates work covered by the following copyright and permission notice:
// SPDX-FileCopyrightText: Copyright (c) 2024 The Shardy Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --automatic-sharding-pipeline="mesh-shape=1,2" %s > %t.mlir
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


func.func @reduce_window(%arg0: tensor<48x48x3xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}, {}]>}, %arg1: tensor<48x48x3xi32>, %arg2: tensor<f32>, %arg3: tensor<i32>) -> (tensor<16x48x1xf32>) {
  %0:2 = "stablehlo.reduce_window"(%arg0, %arg1, %arg2, %arg3) ({
  ^bb0(%arg4: tensor<f32>, %arg5 : tensor<i32>, %arg6: tensor<f32>, %arg7 : tensor<i32>):
    %1 = stablehlo.maximum %arg4, %arg6 : tensor<f32>
    %2 = stablehlo.maximum %arg5, %arg7 : tensor<i32>
    stablehlo.return %1, %2 : tensor<f32>, tensor<i32>
  }) {window_dimensions = array<i64: 3, 1, 3>,
      window_strides = array<i64: 3, 1, 3>,
      padding = dense<[[0, 0], [2, -2], [0, 0]]> : tensor<3x2xi64>}
      : (tensor<48x48x3xf32>, tensor<48x48x3xi32>, tensor<f32>, tensor<i32>) -> (tensor<16x48x1xf32>, tensor<16x48x1xi32>)
  func.return %0#0 : tensor<16x48x1xf32>
}

// CHECK: sdy.manual_computation(%arg0, %arg1, %arg2, %arg3) in_shardings=[<@mesh, [{"batch"}, {}, {}]>, <@mesh, [{"batch"}, {}, {}]>, <@mesh, []>, <@mesh, []>] out_shardings=[<@mesh, [{"batch"}, {}, {}]>]
// CHECK: %2 = stablehlo.maximum %arg8, %arg10 : tensor<f32>
// CHECK: %3 = stablehlo.maximum %arg9, %arg11 : tensor<i32>
// CHECK: stablehlo.return %2, %3 : tensor<f32>, tensor<i32>
// CHECK: sdy.return %1#0 : tensor<8x48x1xf32>
