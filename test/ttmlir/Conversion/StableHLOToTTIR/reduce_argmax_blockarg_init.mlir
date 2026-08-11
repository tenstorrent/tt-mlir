// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// REQUIRES: stablehlo
// RUN: not ttmlir-opt --convert-stablehlo-to-ttir %s 2>&1 | FileCheck %s

// An argmax-shaped reduce whose init value is a block argument (here a
// function argument) rather than a constant. The argmax recognizer used to
// dereference the null result of `getDefiningOp()` and segfault; it must
// report a clean legalization failure instead.
//
// The reducer body has to start with a `compare` op: the reduce pattern
// dispatches on the body's first op, and add/max/min/mul/and/or are claimed by
// earlier branches before the argmax check is reached.

// CHECK: failed to legalize operation 'stablehlo.reduce'
module @reduce_argmax_blockarg_init {
  func.func @main(%init_max: tensor<f64>, %arg0: tensor<2x3xf64>) -> tensor<2xi32> {
    %iota = stablehlo.iota dim = 1 : tensor<2x3xi32>
    %init_idx = stablehlo.constant dense<0> : tensor<i32>
    %r:2 = stablehlo.reduce(%arg0 init: %init_max), (%iota init: %init_idx) across dimensions = [1] : (tensor<2x3xf64>, tensor<2x3xi32>, tensor<f64>, tensor<i32>) -> (tensor<2xf64>, tensor<2xi32>)
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
    return %r#1 : tensor<2xi32>
  }
}
