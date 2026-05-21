// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Test that multiple composites sharing the same decomposition function are
// both inlined correctly and the shared decomposition function is cleaned up.

// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s

// CHECK-LABEL: func.func @test_shared_decomposition
// CHECK: "ttnn.add"
// CHECK: "ttnn.multiply"
// CHECK: "ttnn.add"
// CHECK: "ttnn.multiply"
// CHECK-NOT: "ttcore.composite"
// CHECK-NOT: @shared_decomp
module attributes {} {
  func.func @test_shared_decomposition(
      %arg0: tensor<32x32xbf16>,
      %arg1: tensor<32x32xbf16>,
      %arg2: tensor<32x32xbf16>
  ) -> (tensor<32x32xbf16>, tensor<32x32xbf16>) {
    %0 = "ttcore.composite"(%arg0, %arg1)
        <{composite_name = "custom_op_a",
          decomposition = @shared_decomp,
          composite_attributes = {}}>
        : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %1 = "ttcore.composite"(%arg1, %arg2)
        <{composite_name = "custom_op_b",
          decomposition = @shared_decomp,
          composite_attributes = {}}>
        : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %0, %1 : tensor<32x32xbf16>, tensor<32x32xbf16>
  }

  func.func private @shared_decomp(
      %arg0: tensor<32x32xbf16>,
      %arg1: tensor<32x32xbf16>
  ) -> tensor<32x32xbf16> {
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %1 = "ttir.multiply"(%0, %arg0) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %1 : tensor<32x32xbf16>
  }
}
