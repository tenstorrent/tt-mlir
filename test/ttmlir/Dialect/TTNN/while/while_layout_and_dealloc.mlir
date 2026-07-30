// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

// CHECK-LABEL: func.func @counted

// The counted-loop idiom is recognized, so the runtime can skip the condition
// program and its per-iteration device-to-host synchronization.
// CHECK: ttnn.while
// CHECK-SAME: trip_count = 4 : i64

// The condition is read back to host, so it has to end up as a uint32 tensor
// in system memory.
// CHECK: ttnn.typecast
// CHECK-SAME: -> tensor<ui32
// CHECK: ttnn.from_device
// CHECK-SAME: -> tensor<ui32

// Region block arguments are owned by the caller or by the previous iteration,
// so the loop must never deallocate them. These CHECK-NOTs are bounded by the
// surrounding CHECKs, i.e. they only cover the inside of the regions.
// CHECK-NOT: "ttnn.deallocate"(%arg
// CHECK: ttnn.yield %{{[0-9]+}} : tensor<ui32

// The body's yielded layouts match the block arguments exactly; otherwise
// iteration 2 would disagree with the serialized tensor descriptors. The op
// verifier enforces it, so reaching the end of the pipeline is the proof.
// CHECK: ^bb0(
// CHECK-NOT: "ttnn.deallocate"(%arg
// CHECK: ttnn.yield %{{[0-9]+}}, %{{[0-9]+}} : tensor<si32
// CHECK: } -> (tensor<si32

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
