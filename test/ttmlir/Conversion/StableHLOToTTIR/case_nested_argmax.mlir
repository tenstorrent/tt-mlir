// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// REQUIRES: stablehlo
// RUN: ttmlir-opt --convert-stablehlo-to-ttir %s | FileCheck %s

// An argmax nested inside a case branch, reading its -inf/0 init values from
// constants defined outside the op.
//
// `ttir.case` is IsolatedFromAbove, so the branch cannot reference those
// constants directly and they become captures - block arguments, whose defining
// op the argmax recognizer cannot see. It has to follow the block argument back
// out to the operand it is bound to, or it would decline the reduce and fail the
// whole conversion. Every block argument of a branch is a capture, so unlike a
// while loop there is no init offset to skip.

// CHECK-LABEL: func.func @main
// CHECK: ttir.case
// The constants stay outside the branches, passed in as captures.
// CHECK-SAME: captures(
// CHECK-NOT: ttir.constant
// CHECK: ttir.argmax
module @case_nested_argmax {
  func.func @main(%arg0: tensor<2x3xf64>, %index: tensor<i32>) -> tensor<2xi32> {
    %neg_inf = stablehlo.constant dense<0xFFF0000000000000> : tensor<f64>
    %zero_idx = stablehlo.constant dense<0> : tensor<i32>
    %fallback = stablehlo.constant dense<0> : tensor<2xi32>
    %0 = "stablehlo.case"(%index) ({
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
      stablehlo.return %r#1 : tensor<2xi32>
    }, {
      stablehlo.return %fallback : tensor<2xi32>
    }) : (tensor<i32>) -> tensor<2xi32>
    return %0 : tensor<2xi32>
  }
}
