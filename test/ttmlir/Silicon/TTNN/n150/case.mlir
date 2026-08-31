// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

// Each branch becomes its own private flatbuffer program that the runtime
// executes with a nested ProgramExecutor, after reading the index back to host.
//
// The indices below are constants rather than runtime values, because this
// flatbuffer is executed by a bulk `ttrt run` over the whole Silicon directory,
// which fills inputs with randn - a runtime index would be an arbitrary int32
// and every function would exercise the out-of-range path.

// CHECK-LABEL: func.func @two_branch
// CHECK: ttnn.case
func.func @two_branch(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %index = "ttir.constant"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
  %one = "ttir.constant"() <{value = dense<1.000000e+00> : tensor<32x32xf32>}> : () -> tensor<32x32xf32>
  %r = ttir.case index(%index : tensor<i32>) captures(%arg0, %one : tensor<32x32xf32>, tensor<32x32xf32>)
  branches {
  ^bb0(%a: tensor<32x32xf32>, %b: tensor<32x32xf32>):
    %0 = "ttir.add"(%a, %b) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    ttir.yield %0 : tensor<32x32xf32>
  }, {
  ^bb0(%a: tensor<32x32xf32>, %b: tensor<32x32xf32>):
    %0 = "ttir.subtract"(%a, %b) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    ttir.yield %0 : tensor<32x32xf32>
  } -> (tensor<32x32xf32>)
  return %r : tensor<32x32xf32>
}

// CHECK-LABEL: func.func @three_branch
// CHECK: ttnn.case
func.func @three_branch(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %index = "ttir.constant"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
  %two = "ttir.constant"() <{value = dense<2.000000e+00> : tensor<32x32xf32>}> : () -> tensor<32x32xf32>
  %r = ttir.case index(%index : tensor<i32>) captures(%arg0, %two : tensor<32x32xf32>, tensor<32x32xf32>)
  branches {
  ^bb0(%a: tensor<32x32xf32>, %b: tensor<32x32xf32>):
    %0 = "ttir.multiply"(%a, %b) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    ttir.yield %0 : tensor<32x32xf32>
  }, {
  ^bb0(%a: tensor<32x32xf32>, %b: tensor<32x32xf32>):
    %0 = "ttir.subtract"(%a, %b) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    ttir.yield %0 : tensor<32x32xf32>
  }, {
  ^bb0(%a: tensor<32x32xf32>, %b: tensor<32x32xf32>):
    %0 = "ttir.add"(%a, %b) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    ttir.yield %0 : tensor<32x32xf32>
  } -> (tensor<32x32xf32>)
  return %r : tensor<32x32xf32>
}

// An index past the end selects the last branch, so this must return
// %arg0 * %arg0 rather than trapping or picking branch 0.
// CHECK-LABEL: func.func @out_of_range
// CHECK: ttnn.case
func.func @out_of_range(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %index = "ttir.constant"() <{value = dense<7> : tensor<i32>}> : () -> tensor<i32>
  %r = ttir.case index(%index : tensor<i32>) captures(%arg0 : tensor<32x32xf32>)
  branches {
  ^bb0(%a: tensor<32x32xf32>):
    %0 = "ttir.add"(%a, %a) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    ttir.yield %0 : tensor<32x32xf32>
  }, {
  ^bb0(%a: tensor<32x32xf32>):
    %0 = "ttir.subtract"(%a, %a) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    ttir.yield %0 : tensor<32x32xf32>
  }, {
  ^bb0(%a: tensor<32x32xf32>):
    %0 = "ttir.multiply"(%a, %a) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    ttir.yield %0 : tensor<32x32xf32>
  } -> (tensor<32x32xf32>)
  return %r : tensor<32x32xf32>
}

// A negative index is out of range too, which is why the index stays si32 all
// the way to the runtime rather than being cast to unsigned.
// CHECK-LABEL: func.func @negative_index
// CHECK: ttnn.case
func.func @negative_index(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %index = "ttir.constant"() <{value = dense<-1> : tensor<i32>}> : () -> tensor<i32>
  %r = ttir.case index(%index : tensor<i32>) captures(%arg0 : tensor<32x32xf32>)
  branches {
  ^bb0(%a: tensor<32x32xf32>):
    %0 = "ttir.add"(%a, %a) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    ttir.yield %0 : tensor<32x32xf32>
  }, {
  ^bb0(%a: tensor<32x32xf32>):
    %0 = "ttir.multiply"(%a, %a) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    ttir.yield %0 : tensor<32x32xf32>
  } -> (tensor<32x32xf32>)
  return %r : tensor<32x32xf32>
}

// A branch that yields a capture verbatim: the branch program hands back the
// very tensor it was given, so the runtime has to publish the source rather than
// the retained view it passed in, or the enclosing program could never free it.
// CHECK-LABEL: func.func @identity_branch
// CHECK: ttnn.case
func.func @identity_branch(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %index = "ttir.constant"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
  %r = ttir.case index(%index : tensor<i32>) captures(%arg0 : tensor<32x32xf32>)
  branches {
  ^bb0(%a: tensor<32x32xf32>):
    ttir.yield %a : tensor<32x32xf32>
  }, {
  ^bb0(%a: tensor<32x32xf32>):
    %0 = "ttir.add"(%a, %a) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    ttir.yield %0 : tensor<32x32xf32>
  } -> (tensor<32x32xf32>)
  return %r : tensor<32x32xf32>
}

// A case inside a while body, so the nested ProgramExecutor recursion is
// exercised in both directions.
// CHECK-LABEL: func.func @nested
// CHECK: ttnn.while
// CHECK: ttnn.case
func.func @nested(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %i0 = "ttir.constant"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
  %limit = "ttir.constant"() <{value = dense<3> : tensor<i32>}> : () -> tensor<i32>
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
      // The induction variable picks the branch, so both arms run across the
      // three iterations.
      %acc2 = ttir.case index(%i : tensor<i32>) captures(%acc : tensor<32x32xf32>)
      branches {
      ^bb0(%a: tensor<32x32xf32>):
        %0 = "ttir.add"(%a, %a) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
        ttir.yield %0 : tensor<32x32xf32>
      }, {
      ^bb0(%a: tensor<32x32xf32>):
        %0 = "ttir.multiply"(%a, %a) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
        ttir.yield %0 : tensor<32x32xf32>
      } -> (tensor<32x32xf32>)
      ttir.yield %next, %acc2 : tensor<i32>, tensor<32x32xf32>
    } -> (tensor<i32>, tensor<32x32xf32>)
  return %r#1 : tensor<32x32xf32>
}
