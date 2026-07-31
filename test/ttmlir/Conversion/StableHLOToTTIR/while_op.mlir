// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module @jit_while {
  // The body reads %arg0, %limit and %step from the enclosing scope. ttir.while
  // is IsolatedFromAbove, so those have to be materialized inside the regions:
  // %arg0 is promoted to an explicit capture, while the %limit and %step
  // constants are cloned in (each region gets only the ones it uses).
  //
  // CHECK-LABEL: func.func public @counted
  // CHECK: ttir.while
  // CHECK-SAME: inits(
  // CHECK-SAME: captures(%{{.*}} : tensor<32x32xf32>)
  // The regions must have identical signatures: 2 loop-carried values plus the
  // single non-constant capture.
  // CHECK: cond {
  // CHECK-NEXT: ^bb0(%{{.*}}: tensor<i32>, %{{.*}}: tensor<32x32xf32>, %{{.*}}: tensor<32x32xf32>)
  // %limit is cloned into cond rather than passed in.
  // CHECK: ttir.constant{{.*}}dense<4>
  // CHECK: ttir.yield
  // CHECK: do {
  // CHECK-NEXT: ^bb0(%{{.*}}: tensor<i32>, %{{.*}}: tensor<32x32xf32>, %{{.*}}: tensor<32x32xf32>)
  // ... and %step into the body.
  // CHECK: ttir.constant{{.*}}dense<1>
  // No stablehlo may survive inside the regions.
  // CHECK-NOT: stablehlo.
  // CHECK: ttir.yield
  func.func public @counted(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %start = stablehlo.constant dense<0> : tensor<i32>
    %limit = stablehlo.constant dense<4> : tensor<i32>
    %step = stablehlo.constant dense<1> : tensor<i32>
    %0:2 = stablehlo.while(%iterArg = %start, %iterArg_acc = %arg0) : tensor<i32>, tensor<32x32xf32>
      cond {
        %p = stablehlo.compare LT, %iterArg, %limit, SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
        stablehlo.return %p : tensor<i1>
      } do {
        %next = stablehlo.add %iterArg, %step : tensor<i32>
        %acc = stablehlo.add %iterArg_acc, %arg0 : tensor<32x32xf32>
        stablehlo.return %next, %acc : tensor<i32>, tensor<32x32xf32>
      }
    return %0#1 : tensor<32x32xf32>
  }
}
