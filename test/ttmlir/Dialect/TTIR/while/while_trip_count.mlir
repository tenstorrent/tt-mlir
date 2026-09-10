// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-while-trip-count -o %t %s
// RUN: FileCheck %s --input-file=%t

// A counted loop: constant init, comparison against a loop-invariant constant,
// constant step. 0, 1, 2, 3 -> 4 iterations.
// CHECK-LABEL: func.func @counted
// CHECK: ttir.while
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

// Canonicalization commutes comparisons, so the induction variable shows up on
// the right-hand side. 2, 4, 6, 8, 10 -> 5 iterations (inclusive bound).
// CHECK-LABEL: func.func @commuted_and_inclusive
// CHECK: ttir.while
// CHECK-SAME: trip_count = 5 : i64
func.func @commuted_and_inclusive(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %i0 = "ttir.constant"() <{value = dense<2> : tensor<i32>}> : () -> tensor<i32>
  %limit = "ttir.constant"() <{value = dense<10> : tensor<i32>}> : () -> tensor<i32>
  %step = "ttir.constant"() <{value = dense<2> : tensor<i32>}> : () -> tensor<i32>
  %r:2 = ttir.while inits(%i0, %arg0 : tensor<i32>, tensor<32x32xf32>)
                    captures(%limit, %step : tensor<i32>, tensor<i32>)
    cond {
    ^cond(%i: tensor<i32>, %acc: tensor<32x32xf32>, %l: tensor<i32>, %s: tensor<i32>):
      %p = "ttir.ge"(%l, %i) : (tensor<i32>, tensor<i32>) -> tensor<i1>
      ttir.yield %p : tensor<i1>
    } do {
    ^body(%i: tensor<i32>, %acc: tensor<32x32xf32>, %l: tensor<i32>, %s: tensor<i32>):
      %next = "ttir.add"(%i, %s) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      ttir.yield %next, %acc : tensor<i32>, tensor<32x32xf32>
    } -> (tensor<i32>, tensor<32x32xf32>)
  return %r#1 : tensor<32x32xf32>
}

// The limit is a runtime value, so the loop stays data-dependent.
// CHECK-LABEL: func.func @data_dependent
// CHECK: ttir.while
// CHECK-NOT: trip_count
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
      ttir.yield %next, %acc : tensor<i32>, tensor<32x32xf32>
    } -> (tensor<i32>, tensor<32x32xf32>)
  return %r#1 : tensor<32x32xf32>
}

// The step moves away from the limit, so the loop is not provably counted and
// must be left alone rather than annotated with a bogus count.
// CHECK-LABEL: func.func @diverging_step
// CHECK: ttir.while
// CHECK-NOT: trip_count
func.func @diverging_step(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %i0 = "ttir.constant"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
  %limit = "ttir.constant"() <{value = dense<4> : tensor<i32>}> : () -> tensor<i32>
  %step = "ttir.constant"() <{value = dense<-1> : tensor<i32>}> : () -> tensor<i32>
  %r:2 = ttir.while inits(%i0, %arg0 : tensor<i32>, tensor<32x32xf32>)
                    captures(%limit, %step : tensor<i32>, tensor<i32>)
    cond {
    ^cond(%i: tensor<i32>, %acc: tensor<32x32xf32>, %l: tensor<i32>, %s: tensor<i32>):
      %p = "ttir.lt"(%i, %l) : (tensor<i32>, tensor<i32>) -> tensor<i1>
      ttir.yield %p : tensor<i1>
    } do {
    ^body(%i: tensor<i32>, %acc: tensor<32x32xf32>, %l: tensor<i32>, %s: tensor<i32>):
      %next = "ttir.add"(%i, %s) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      ttir.yield %next, %acc : tensor<i32>, tensor<32x32xf32>
    } -> (tensor<i32>, tensor<32x32xf32>)
  return %r#1 : tensor<32x32xf32>
}
