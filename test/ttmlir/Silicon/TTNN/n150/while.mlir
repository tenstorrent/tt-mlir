// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

// Each region becomes its own private flatbuffer program that the runtime
// executes with a nested ProgramExecutor.

// A counted loop: the runtime uses trip_count and never runs the condition
// program, so there is no per-iteration device-to-host synchronization.
// CHECK-LABEL: func.func @counted
// CHECK: ttnn.while
// CHECK-SAME: trip_count = 4 : i64
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

// A data-dependent loop: the limit is a runtime value, so the condition
// program runs every iteration and its result is read back to host.
//
// The limit is clamped to at most 8 so the loop terminates for any value of
// %arg1. That matters because this flatbuffer is executed by a bulk `ttrt run`
// over the whole Silicon directory, which fills inputs with randn -- an
// unclamped runtime limit would be an arbitrary int32 and the loop would run
// until the runtime's iteration backstop. Clamping keeps the limit a runtime
// value, so the condition program still runs every iteration.
// CHECK-LABEL: func.func @data_dependent
// CHECK: ttnn.while
// CHECK-NOT: trip_count
func.func @data_dependent(%arg0: tensor<32x32xf32>, %arg1: tensor<i32>) -> tensor<32x32xf32> {
  %i0 = "ttir.constant"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
  %step = "ttir.constant"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
  %max = "ttir.constant"() <{value = dense<8> : tensor<i32>}> : () -> tensor<i32>
  %limit = "ttir.minimum"(%arg1, %max) : (tensor<i32>, tensor<i32>) -> tensor<i32>
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

// The body yields one of its block arguments unchanged. The nested program's
// output resolves straight to the input slot it was handed, so the loop gets
// back the very retained view it passed in and has to map it to the tensor the
// view was made from -- publishing a view as a loop result would leave it
// retained and stop the enclosing program from ever freeing it.
// CHECK-LABEL: func.func @passthrough
// CHECK: ttnn.while
// CHECK-SAME: trip_count = 3 : i64
func.func @passthrough(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>)
    -> (tensor<32x32xf32>, tensor<32x32xf32>) {
  %i0 = "ttir.constant"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
  %limit = "ttir.constant"() <{value = dense<3> : tensor<i32>}> : () -> tensor<i32>
  %step = "ttir.constant"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
  %r:3 = ttir.while inits(%i0, %arg0, %arg1 : tensor<i32>, tensor<32x32xf32>, tensor<32x32xf32>)
                    captures(%limit, %step : tensor<i32>, tensor<i32>)
    cond {
    ^cond(%i: tensor<i32>, %a: tensor<32x32xf32>, %b: tensor<32x32xf32>, %l: tensor<i32>, %s: tensor<i32>):
      %p = "ttir.lt"(%i, %l) : (tensor<i32>, tensor<i32>) -> tensor<i1>
      ttir.yield %p : tensor<i1>
    } do {
    ^body(%i: tensor<i32>, %a: tensor<32x32xf32>, %b: tensor<32x32xf32>, %l: tensor<i32>, %s: tensor<i32>):
      %next = "ttir.add"(%i, %s) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      %a2 = "ttir.add"(%a, %b) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
      // %b is carried through untouched.
      ttir.yield %next, %a2, %b : tensor<i32>, tensor<32x32xf32>, tensor<32x32xf32>
    } -> (tensor<i32>, tensor<32x32xf32>, tensor<32x32xf32>)
  return %r#1, %r#2 : tensor<32x32xf32>, tensor<32x32xf32>
}

// Nested loops: a while inside a while body. The inner loop's regions become
// additional programs, and the runtime's ProgramExecutor recursion handles the
// nesting with no depth limit.
// CHECK-LABEL: func.func @nested
// CHECK: ttnn.while
// CHECK: ttnn.while
func.func @nested(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %i0 = "ttir.constant"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
  %limit = "ttir.constant"() <{value = dense<2> : tensor<i32>}> : () -> tensor<i32>
  %step = "ttir.constant"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
  %r:2 = ttir.while inits(%i0, %arg0 : tensor<i32>, tensor<32x32xf32>)
                    captures(%limit, %step : tensor<i32>, tensor<i32>)
    cond {
    ^cond(%i: tensor<i32>, %acc: tensor<32x32xf32>, %l: tensor<i32>, %s: tensor<i32>):
      %p = "ttir.lt"(%i, %l) : (tensor<i32>, tensor<i32>) -> tensor<i1>
      ttir.yield %p : tensor<i1>
    } do {
    ^body(%i: tensor<i32>, %acc: tensor<32x32xf32>, %l: tensor<i32>, %s: tensor<i32>):
      %inner:2 = ttir.while inits(%i, %acc : tensor<i32>, tensor<32x32xf32>)
                            captures(%l, %s : tensor<i32>, tensor<i32>)
        cond {
        ^icond(%j: tensor<i32>, %iacc: tensor<32x32xf32>, %il: tensor<i32>, %is: tensor<i32>):
          %ip = "ttir.lt"(%j, %il) : (tensor<i32>, tensor<i32>) -> tensor<i1>
          ttir.yield %ip : tensor<i1>
        } do {
        ^ibody(%j: tensor<i32>, %iacc: tensor<32x32xf32>, %il: tensor<i32>, %is: tensor<i32>):
          %jnext = "ttir.add"(%j, %is) : (tensor<i32>, tensor<i32>) -> tensor<i32>
          %iacc2 = "ttir.add"(%iacc, %iacc) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
          ttir.yield %jnext, %iacc2 : tensor<i32>, tensor<32x32xf32>
        } -> (tensor<i32>, tensor<32x32xf32>)
      %next = "ttir.add"(%i, %s) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      ttir.yield %next, %inner#1 : tensor<i32>, tensor<32x32xf32>
    } -> (tensor<i32>, tensor<32x32xf32>)
  return %r#1 : tensor<32x32xf32>
}
