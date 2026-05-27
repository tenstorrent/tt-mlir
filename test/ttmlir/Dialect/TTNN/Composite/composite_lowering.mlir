// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Test that with OpModel enabled, ttcore.composite is resolved to the typed
// ttnn.topk_router_gpt op through the backend pipeline.

// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s

// CHECK-LABEL: func.func @test_composite_lowering
// CHECK: "ttnn.topk_router_gpt"
// CHECK-NOT: "ttcore.composite"
// CHECK-NOT: @topk_router_gpt_decomp
module attributes {} {
  func.func @test_composite_lowering(
      %input:  tensor<32x64xbf16>,
      %weight: tensor<64x128xbf16>,
      %bias:   tensor<32x128xbf16>
  ) -> (tensor<32x4xui16>, tensor<32x4xbf16>) {
    %indices, %weights = "ttcore.composite"(%input, %weight, %bias)
        <{composite_name = "topk_router_gpt",
          decomposition = @topk_router_gpt_decomp,
          composite_attributes = {k = 4 : i32, num_experts = 128 : i32}}>
        : (tensor<32x64xbf16>, tensor<64x128xbf16>, tensor<32x128xbf16>)
          -> (tensor<32x4xui16>, tensor<32x4xbf16>)
    return %indices, %weights : tensor<32x4xui16>, tensor<32x4xbf16>
  }

  func.func private @topk_router_gpt_decomp(
      %arg0: tensor<32x64xbf16>,
      %arg1: tensor<64x128xbf16>,
      %arg2: tensor<32x128xbf16>
  ) -> (tensor<32x4xui16>, tensor<32x4xbf16>) {
    %0:2 = "ttir.topk_router_gpt"(%arg0, %arg1, %arg2)
        <{k = 4 : i32, num_experts = 128 : i32}>
        : (tensor<32x64xbf16>, tensor<64x128xbf16>, tensor<32x128xbf16>)
          -> (tensor<32x4xui16>, tensor<32x4xbf16>)
    return %0#0, %0#1 : tensor<32x4xui16>, tensor<32x4xbf16>
  }
}
