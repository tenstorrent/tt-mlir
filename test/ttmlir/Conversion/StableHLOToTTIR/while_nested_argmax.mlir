// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// REQUIRES: stablehlo
// RUN: ttmlir-opt --convert-stablehlo-to-ttir %s | FileCheck %s

// An argmax nested inside a while loop, reading its -inf/0 init values from
// constants defined outside the loop (the shape LU pivoting lowers to).
//
// `ttir.while` is IsolatedFromAbove, so the region cannot reference those
// constants directly. They must be cloned into the region rather than promoted
// to captures: as block arguments they are invisible to the argmax recognizer,
// which then declines the reduce and fails the whole conversion.
//
// %arg0 is a genuine non-constant capture, so both paths are exercised.

// CHECK-LABEL: func.func @main
// CHECK: ttir.while
// The -inf and 0 init values are cloned into the region, not passed in.
// CHECK: ttir.constant
// CHECK: ttir.argmax
module @while_nested_argmax {
  func.func @main(%arg0: tensor<2x3xf64>) -> tensor<2xi32> {
    %iter_init = stablehlo.constant dense<0> : tensor<i64>
    %limit = stablehlo.constant dense<3> : tensor<i64>
    %one = stablehlo.constant dense<1> : tensor<i64>
    %neg_inf = stablehlo.constant dense<0xFFF0000000000000> : tensor<f64>
    %zero_idx = stablehlo.constant dense<0> : tensor<i32>
    %idx_init = stablehlo.constant dense<0> : tensor<2xi32>
    %0:2 = stablehlo.while(%iterArg = %iter_init, %iterArg_0 = %idx_init) : tensor<i64>, tensor<2xi32>
     cond {
      %1 = stablehlo.compare  LT, %iterArg, %limit,  SIGNED : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %1 : tensor<i1>
    } do {
      %iota = stablehlo.iota dim = 1 : tensor<2x3xi32>
      %r:2 = stablehlo.reduce(%arg0 init: %neg_inf), (%iota init: %zero_idx) across dimensions = [1] : (tensor<2x3xf64>, tensor<2x3xi32>, tensor<f64>, tensor<i32>) -> (tensor<2xf64>, tensor<2xi32>)
       reducer(%acc_val: tensor<f64>, %val: tensor<f64>) (%acc_idx: tensor<i32>, %idx: tensor<i32>)  {
        %gt = stablehlo.compare  GT, %acc_val, %val,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
        %ne = stablehlo.compare  NE, %acc_val, %acc_val,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
        %take_val = stablehlo.or %gt, %ne : tensor<i1>
        %eq = stablehlo.compare  EQ, %acc_val, %val,  FLOAT : (tensor<f64>, tensor<f64>) -> tensor<i1>
        %lt = stablehlo.compare  LT, %acc_idx, %idx,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
        %tie = stablehlo.and %eq, %lt : tensor<i1>
        %take_idx = stablehlo.or %take_val, %tie : tensor<i1>
        %out_val = stablehlo.select %take_val, %acc_val, %val : tensor<i1>, tensor<f64>
        %out_idx = stablehlo.select %take_idx, %acc_idx, %idx : tensor<i1>, tensor<i32>
        stablehlo.return %out_val, %out_idx : tensor<f64>, tensor<i32>
      }
      %next = stablehlo.add %iterArg, %one : tensor<i64>
      stablehlo.return %next, %r#1 : tensor<i64>, tensor<2xi32>
    }
    return %0#1 : tensor<2xi32>
  }
}
