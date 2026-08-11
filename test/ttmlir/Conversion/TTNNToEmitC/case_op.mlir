// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-to-ttnn-common-pipeline="enable-const-eval=false" -o %t.mlir %s
// RUN: ttmlir-opt --ttnn-common-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: FileCheck %s --input-file=%t2.mlir

// A case becomes an `emitc.switch`. The values the branches produce live in
// emitc.variable lvalues because emitc.switch has no results, and the index is
// read out of the tensor into a scalar first because the switch selects on one.
//
// The last branch is the `default` arm, which is what reproduces the op's rule
// that an out-of-range index selects it.
//
// CHECK-LABEL: func.func @three_branch
// CHECK: emitc.variable
// CHECK-SAME: #emitc.opaque<"::ttnn::Tensor()">
// CHECK: emitc.variable
// CHECK-SAME: -> !emitc.lvalue<!emitc.opaque<"int32_t">>
// CHECK: emitc.verbatim "{}{{.*}}to_vector<int32_t>()[0];"
// CHECK: emitc.switch %{{[0-9]+}} : !emitc.opaque<"int32_t">
// Inside a region the printer elides the dialect prefix.
// CHECK: case 0 {
// CHECK: assign %{{[0-9]+}} : !emitc.opaque<"::ttnn::Tensor"> to
// CHECK: case 1 {
// CHECK: assign %{{[0-9]+}} : !emitc.opaque<"::ttnn::Tensor"> to
// CHECK: default {
// CHECK: assign %{{[0-9]+}} : !emitc.opaque<"::ttnn::Tensor"> to
// CHECK: emitc.load
func.func @three_branch(%arg0: tensor<32x32xf32>, %index: tensor<i32>) -> tensor<32x32xf32> {
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

// With two branches there is a single case label and the default arm; with one
// there would be only the default.
// CHECK-LABEL: func.func @two_branch
// CHECK: emitc.switch
// CHECK: case 0 {
// CHECK-NOT: case 1 {
// CHECK: default {
func.func @two_branch(%arg0: tensor<32x32xf32>, %index: tensor<i32>) -> tensor<32x32xf32> {
  %r = ttir.case index(%index : tensor<i32>) captures(%arg0 : tensor<32x32xf32>)
  branches {
  ^bb0(%a: tensor<32x32xf32>):
    %0 = "ttir.abs"(%a) : (tensor<32x32xf32>) -> tensor<32x32xf32>
    ttir.yield %0 : tensor<32x32xf32>
  }, {
  ^bb0(%a: tensor<32x32xf32>):
    %0 = "ttir.neg"(%a) : (tensor<32x32xf32>) -> tensor<32x32xf32>
    ttir.yield %0 : tensor<32x32xf32>
  } -> (tensor<32x32xf32>)
  return %r : tensor<32x32xf32>
}
