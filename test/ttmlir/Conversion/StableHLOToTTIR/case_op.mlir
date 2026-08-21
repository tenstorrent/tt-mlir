// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module @jit_case {
  // This is what `jax.lax.cond` lowers to: the branches take no block arguments
  // at all and read %arg0 straight from the enclosing scope. ttir.case is
  // IsolatedFromAbove, so that has to be promoted to an explicit capture and
  // become every branch's block argument.
  //
  // CHECK-LABEL: func.func public @two_branch
  // CHECK: ttir.case
  // CHECK-SAME: index(
  // CHECK-SAME: captures(
  // Both branches take exactly the captures, in the same order.
  // CHECK: branches {
  // CHECK-NEXT: ^bb0(%{{.*}}: tensor<f32>)
  // No stablehlo may survive inside the branches.
  // CHECK-NOT: stablehlo.
  // CHECK: ttir.yield
  // CHECK: ^bb0(%{{.*}}: tensor<f32>)
  // CHECK-NOT: stablehlo.
  // CHECK: ttir.yield
  func.func public @two_branch(%arg0: tensor<f32>) -> tensor<f32> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.compare GT, %arg0, %cst, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %index = stablehlo.convert %0 : (tensor<i1>) -> tensor<i32>
    %1 = "stablehlo.case"(%index) ({
      %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %2 = stablehlo.add %arg0, %cst_0 : tensor<f32>
      stablehlo.return %2 : tensor<f32>
    }, {
      %cst_0 = stablehlo.constant dense<1.000000e+00> : tensor<f32>
      %2 = stablehlo.subtract %arg0, %cst_0 : tensor<f32>
      stablehlo.return %2 : tensor<f32>
    }) : (tensor<i32>) -> tensor<f32>
    return %1 : tensor<f32>
  }

  // And this is `jax.lax.switch`: three branches, still no block arguments. The
  // constant lives outside, so the captures are %arg0 and it.
  //
  // CHECK-LABEL: func.func public @three_branch
  // CHECK: ttir.case
  // CHECK-SAME: captures(%{{.*}}, %{{.*}} : tensor<32x32xf32>, tensor<32x32xf32>)
  // CHECK-COUNT-3: ttir.yield
  // CHECK-NOT: stablehlo.
  func.func public @three_branch(%arg0: tensor<32x32xf32>, %index: tensor<i32>) -> tensor<32x32xf32> {
    %two = stablehlo.constant dense<2.000000e+00> : tensor<32x32xf32>
    %0 = "stablehlo.case"(%index) ({
      %1 = stablehlo.multiply %arg0, %two : tensor<32x32xf32>
      stablehlo.return %1 : tensor<32x32xf32>
    }, {
      %1 = stablehlo.subtract %arg0, %two : tensor<32x32xf32>
      stablehlo.return %1 : tensor<32x32xf32>
    }, {
      %1 = stablehlo.add %arg0, %two : tensor<32x32xf32>
      stablehlo.return %1 : tensor<32x32xf32>
    }) : (tensor<i32>) -> tensor<32x32xf32>
    return %0 : tensor<32x32xf32>
  }
}
