// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

// CHECK-LABEL: func.func @three_branch

// The index is read back to host to pick a branch, so it has to end up as an
// si32 tensor in system memory. It stays signed: a negative index has to keep
// reading as out of range, which selects the last branch.
// CHECK: ttnn.from_device
// CHECK-SAME: -> tensor<si32
// CHECK: ttnn.case index(%{{[0-9]+}} : tensor<si32

// Branch block arguments are the captures, owned by the caller, so a branch must
// never deallocate them. These CHECK-NOTs are bounded by the surrounding
// CHECKs, i.e. they only cover the inside of the branches.
// CHECK-NOT: "ttnn.deallocate"(%arg
// CHECK: ttnn.yield %{{[0-9]+}} : tensor<32x32xf32
// CHECK: ^bb0(
// CHECK-NOT: "ttnn.deallocate"(%arg
// CHECK: ttnn.yield %{{[0-9]+}} : tensor<32x32xf32
// CHECK: ^bb0(
// CHECK-NOT: "ttnn.deallocate"(%arg

// Every branch yields the op's result type exactly; otherwise the consumer would
// read one descriptor for values that disagree. The op verifier enforces it, so
// reaching the end of the pipeline is the proof.
// CHECK: } -> (tensor<32x32xf32

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

// A branch whose last op is a reshape would otherwise yield a row-major tensor
// while the results are tiled, which the layout rewriter has to reconcile.
// CHECK-LABEL: func.func @branches_disagree_on_layout
// CHECK: ttnn.case
// CHECK: } -> (tensor<32x32xf32
func.func @branches_disagree_on_layout(%arg0: tensor<32x32xf32>, %flat: tensor<1024xf32>, %index: tensor<i32>) -> tensor<32x32xf32> {
  %r = ttir.case index(%index : tensor<i32>) captures(%arg0, %flat : tensor<32x32xf32>, tensor<1024xf32>)
  branches {
  ^bb0(%a: tensor<32x32xf32>, %f: tensor<1024xf32>):
    %0 = "ttir.add"(%a, %a) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    ttir.yield %0 : tensor<32x32xf32>
  }, {
  ^bb0(%a: tensor<32x32xf32>, %f: tensor<1024xf32>):
    %0 = "ttir.reshape"(%f) <{shape = [32 : i32, 32 : i32]}> : (tensor<1024xf32>) -> tensor<32x32xf32>
    ttir.yield %0 : tensor<32x32xf32>
  } -> (tensor<32x32xf32>)
  return %r : tensor<32x32xf32>
}
