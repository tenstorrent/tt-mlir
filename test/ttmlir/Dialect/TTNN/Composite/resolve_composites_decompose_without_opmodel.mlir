// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Test that without OpModel, a composite named "topk_router_gpt" (which is in
// the registry) still falls back to inlining the decomposition body. The
// decomposition uses trivial ops that don't match topk semantics — we only
// care that the inlined ops appear in place of the composite.

// UNSUPPORTED: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s

// CHECK-LABEL: func.func @test_decompose_without_opmodel
// CHECK: "ttnn.add"
// CHECK: "ttnn.multiply"
// CHECK-NOT: "ttcore.composite"
// CHECK-NOT: "ttnn.topk_router_gpt"
// CHECK-NOT: @topk_router_gpt_decomp
module attributes {} {
  func.func @test_decompose_without_opmodel(
      %arg0: tensor<32x32xbf16>,
      %arg1: tensor<32x32xbf16>,
      %arg2: tensor<32x32xbf16>
  ) -> (tensor<32x32xbf16>, tensor<32x32xbf16>) {
    %0, %1 = "ttcore.composite"(%arg0, %arg1, %arg2)
        <{composite_name = "topk_router_gpt",
          decomposition = @topk_router_gpt_decomp,
          composite_attributes = {k = 4 : i32, num_experts = 128 : i32}}>
        : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>)
          -> (tensor<32x32xbf16>, tensor<32x32xbf16>)
    return %0, %1 : tensor<32x32xbf16>, tensor<32x32xbf16>
  }

  func.func private @topk_router_gpt_decomp(
      %arg0: tensor<32x32xbf16>,
      %arg1: tensor<32x32xbf16>,
      %arg2: tensor<32x32xbf16>
  ) -> (tensor<32x32xbf16>, tensor<32x32xbf16>) {
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %1 = "ttir.multiply"(%0, %arg2) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %0, %1 : tensor<32x32xbf16>, tensor<32x32xbf16>
  }
}
