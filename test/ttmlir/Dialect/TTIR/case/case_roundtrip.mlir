// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --split-input-file %s -o %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir

// `branches` is a variadic region, which no other op in this dialect uses, so
// its parse/print round-trip is worth pinning down on its own.

// CHECK-LABEL: func.func @single_branch
// CHECK: ttir.case index(%{{.*}} : tensor<i32>) captures(%{{.*}} : tensor<1xi32>) branches {
// CHECK: } -> (tensor<1xi32>)
func.func @single_branch(%index: tensor<i32>, %cap: tensor<1xi32>) -> tensor<1xi32> {
  %r = ttir.case index(%index : tensor<i32>) captures(%cap : tensor<1xi32>)
  branches {
  ^bb0(%c: tensor<1xi32>):
    ttir.yield %c : tensor<1xi32>
  } -> (tensor<1xi32>)
  return %r : tensor<1xi32>
}

// -----

// Three branches, so the parser has to keep reading comma-separated regions.
// CHECK-LABEL: func.func @three_branches
// CHECK: ttir.case
// CHECK-COUNT-3: ttir.yield
func.func @three_branches(%index: tensor<i32>, %cap: tensor<1xi32>) -> tensor<1xi32> {
  %r = ttir.case index(%index : tensor<i32>) captures(%cap : tensor<1xi32>)
  branches {
  ^bb0(%c: tensor<1xi32>):
    ttir.yield %c : tensor<1xi32>
  }, {
  ^bb0(%c: tensor<1xi32>):
    %0 = "ttir.abs"(%c) : (tensor<1xi32>) -> tensor<1xi32>
    ttir.yield %0 : tensor<1xi32>
  }, {
  ^bb0(%c: tensor<1xi32>):
    %0 = "ttir.neg"(%c) : (tensor<1xi32>) -> tensor<1xi32>
    ttir.yield %0 : tensor<1xi32>
  } -> (tensor<1xi32>)
  return %r : tensor<1xi32>
}

// -----

// No captures and no results, so both optional clauses are omitted on the parse
// and the print side. The branch omits its terminator too, which
// SingleBlockImplicitTerminator has to insert.
// CHECK-LABEL: func.func @no_captures
// CHECK: ttir.case index(%{{.*}} : tensor<i32>) branches {
// CHECK-NOT: captures(
// CHECK-NOT: ->
func.func @no_captures(%index: tensor<i32>) {
  ttir.case index(%index : tensor<i32>)
  branches {
  ^bb0():
  }
  return
}

// -----

// A case nested in a while body, which is what makes the region-program
// numbering in the serializer worth keeping per op kind.
// CHECK-LABEL: func.func @nested
// CHECK: ttir.while
// CHECK: ttir.case
func.func @nested(%arg0: tensor<1xi32>, %index: tensor<i32>) -> tensor<1xi32> {
  %limit = "ttir.constant"() <{value = dense<4> : tensor<1xi32>}> : () -> tensor<1xi32>
  %r = ttir.while inits(%arg0 : tensor<1xi32>) captures(%limit, %index : tensor<1xi32>, tensor<i32>)
    cond {
    ^cond(%i: tensor<1xi32>, %l: tensor<1xi32>, %idx: tensor<i32>):
      %p = "ttir.lt"(%i, %l) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi1>
      ttir.yield %p : tensor<1xi1>
    } do {
    ^body(%i: tensor<1xi32>, %l: tensor<1xi32>, %idx: tensor<i32>):
      %next = ttir.case index(%idx : tensor<i32>) captures(%i, %l : tensor<1xi32>, tensor<1xi32>)
      branches {
      ^bb0(%a: tensor<1xi32>, %b: tensor<1xi32>):
        %0 = "ttir.add"(%a, %b) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
        ttir.yield %0 : tensor<1xi32>
      }, {
      ^bb0(%a: tensor<1xi32>, %b: tensor<1xi32>):
        %0 = "ttir.subtract"(%a, %b) : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
        ttir.yield %0 : tensor<1xi32>
      } -> (tensor<1xi32>)
      ttir.yield %next : tensor<1xi32>
    } -> (tensor<1xi32>)
  return %r : tensor<1xi32>
}
