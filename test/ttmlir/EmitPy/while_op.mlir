// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-to-ttnn-common-pipeline="enable-const-eval=false" -o %t.mlir %s
// RUN: ttmlir-opt --ttnn-common-to-emitpy-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-python -o %t.py %t2.mlir
// RUN: FileCheck %s --input-file=%t.py

// A counted loop gets an explicit counter, since emitpy.while models a Python
// `while` rather than a `for`. Loop-carried values become named locals that
// the body reassigns.
// CHECK-LABEL: def counted
// CHECK: while_0_carried_0 =
// CHECK: while_0_carried_1 =
// CHECK: while_0_i = 0
// CHECK: while while_0_i < 4:
// CHECK-NOT: break
// CHECK: while_0_carried_0 = ttnn_add
// CHECK: while_0_i = while_0_i + 1
func.func @counted(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %i0 = "ttir.constant"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
  %limit = "ttir.constant"() <{value = dense<4> : tensor<i32>}> : () -> tensor<i32>
  %step = "ttir.constant"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
  %r:2 = ttir.while inits(%i0, %arg0 : tensor<i32>, tensor<32x32xf32>)
                    captures(%limit, %step : tensor<i32>, tensor<i32>)
    cond {
    ^cond(%i: tensor<i32>, %acc: tensor<32x32xf32>, %l: tensor<i32>, %s: tensor<i32>):
      %p = "ttir.lt"(%i, %l) : (tensor<i32>, tensor<i32>) -> tensor<i1>
      ttir.yield %p : tensor<i1>
    } do {
    ^body(%i: tensor<i32>, %acc: tensor<32x32xf32>, %l: tensor<i32>, %s: tensor<i32>):
      %next = "ttir.add"(%i, %s) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      %acc2 = "ttir.add"(%acc, %acc) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
      ttir.yield %next, %acc2 : tensor<i32>, tensor<32x32xf32>
    } -> (tensor<i32>, tensor<32x32xf32>)
  return %r#1 : tensor<32x32xf32>
}

// A data-dependent loop runs the condition every iteration and reads the
// predicate back to host to decide whether to keep going.
// CHECK-LABEL: def data_dependent
// CHECK: while True:
// CHECK: if {{.*}}.to_torch().item() == 0: break
func.func @data_dependent(%arg0: tensor<32x32xf32>, %arg1: tensor<i32>) -> tensor<32x32xf32> {
  %i0 = "ttir.constant"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
  %step = "ttir.constant"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
  %r:2 = ttir.while inits(%i0, %arg0 : tensor<i32>, tensor<32x32xf32>)
                    captures(%arg1, %step : tensor<i32>, tensor<i32>)
    cond {
    ^cond(%i: tensor<i32>, %acc: tensor<32x32xf32>, %l: tensor<i32>, %s: tensor<i32>):
      %p = "ttir.lt"(%i, %l) : (tensor<i32>, tensor<i32>) -> tensor<i1>
      ttir.yield %p : tensor<i1>
    } do {
    ^body(%i: tensor<i32>, %acc: tensor<32x32xf32>, %l: tensor<i32>, %s: tensor<i32>):
      %next = "ttir.add"(%i, %s) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      %acc2 = "ttir.add"(%acc, %acc) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
      ttir.yield %next, %acc2 : tensor<i32>, tensor<32x32xf32>
    } -> (tensor<i32>, tensor<32x32xf32>)
  return %r#1 : tensor<32x32xf32>
}
