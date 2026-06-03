// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Test that a composite with an unknown name (not in the registry) falls
// through to decomposition regardless of OpModel availability.

// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s

// CHECK-LABEL: func.func @test_unknown_composite_name
// CHECK: "ttnn.subtract"
// CHECK: "ttnn.multiply"
// CHECK-NOT: "ttcore.composite"
// CHECK-NOT: @unknown_op_decomp
module attributes {} {
  func.func @test_unknown_composite_name(
      %arg0: tensor<32x32xbf16>,
      %arg1: tensor<32x32xbf16>
  ) -> tensor<32x32xbf16> {
    %0 = "ttcore.composite"(%arg0, %arg1)
        <{composite_name = "unknown_op",
          decomposition = @unknown_op_decomp,
          composite_attributes = {}}>
        : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %0 : tensor<32x32xbf16>
  }

  func.func private @unknown_op_decomp(
      %arg0: tensor<32x32xbf16>,
      %arg1: tensor<32x32xbf16>
  ) -> tensor<32x32xbf16> {
    %0 = "ttir.subtract"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %1 = "ttir.multiply"(%0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %1 : tensor<32x32xbf16>
  }
}
