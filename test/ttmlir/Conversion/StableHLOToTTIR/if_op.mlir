// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// REQUIRES: stablehlo
// RUN: ttmlir-opt --convert-stablehlo-to-ttir %s | FileCheck %s

module @jit_if {
  // `stablehlo.if` runs its true branch when the predicate holds, while
  // ttir.case selects by index, so the false branch has to become branch 0 and
  // the true branch branch 1. The subtract below is the false branch, so it must
  // come first in the lowered op.
  //
  // CHECK-LABEL: func.func public @conditional
  // CHECK: %[[INDEX:[0-9]+]] = "ttir.typecast"{{.*}} -> tensor<i32>
  // CHECK: ttir.case index(%[[INDEX]] :
  // CHECK-SAME: captures(
  // CHECK: branches {
  // CHECK: ttir.subtract
  // CHECK: ttir.yield
  // CHECK: ttir.add
  // CHECK: ttir.yield
  func.func public @conditional(%arg0: tensor<32x32xf32>, %pred: tensor<i1>) -> tensor<32x32xf32> {
    %one = stablehlo.constant dense<1.000000e+00> : tensor<32x32xf32>
    %0 = "stablehlo.if"(%pred) ({
      %1 = stablehlo.add %arg0, %one : tensor<32x32xf32>
      stablehlo.return %1 : tensor<32x32xf32>
    }, {
      %1 = stablehlo.subtract %arg0, %one : tensor<32x32xf32>
      stablehlo.return %1 : tensor<32x32xf32>
    }) : (tensor<i1>) -> tensor<32x32xf32>
    return %0 : tensor<32x32xf32>
  }
}
