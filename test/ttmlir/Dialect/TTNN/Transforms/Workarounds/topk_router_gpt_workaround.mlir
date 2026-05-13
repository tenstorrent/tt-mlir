// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // k=4: k_padded=8, so the workaround must fire and insert two slice_static
  // ops trimming the last dim from 8 down to 4.
  func.func @test_topk_router_gpt_k4_workaround(
      %input:  tensor<32x64xbf16>,
      %weight: tensor<64x128xbf16>,
      %bias:   tensor<32x128xbf16>
  ) -> (tensor<32x4xui16>, tensor<32x4xbf16>) {
    // CHECK-LABEL: func.func @test_topk_router_gpt_k4_workaround
    // CHECK: "ttnn.topk_router_gpt"
    // CHECK-SAME: k = 8 : i32, num_experts = 128 : i32
    // CHECK: "ttnn.slice_static"
    // CHECK-SAME: ends = [32 : i32, 4 : i32]
    // CHECK-SAME: -> tensor<32x4xui16
    // CHECK: "ttnn.slice_static"
    // CHECK-SAME: ends = [32 : i32, 4 : i32]
    // CHECK-SAME: -> tensor<32x4xbf16
    %indices, %weights = "ttir.topk_router_gpt"(%input, %weight, %bias)
        <{k = 4 : i32, num_experts = 128 : i32}>
        : (tensor<32x64xbf16>, tensor<64x128xbf16>, tensor<32x128xbf16>)
          -> (tensor<32x4xui16>, tensor<32x4xbf16>)
    return %indices, %weights : tensor<32x4xui16>, tensor<32x4xbf16>
  }

  // k=8: k_padded=8 == k, so the workaround must NOT fire.
  func.func @test_topk_router_gpt_k8_no_workaround(
      %input:  tensor<32x64xbf16>,
      %weight: tensor<64x128xbf16>,
      %bias:   tensor<32x128xbf16>
  ) -> (tensor<32x8xui16>, tensor<32x8xbf16>) {
    // CHECK-LABEL: func.func @test_topk_router_gpt_k8_no_workaround
    // CHECK: "ttnn.topk_router_gpt"
    // CHECK-NOT: ttnn.slice_static
    // CHECK: return
    %indices, %weights = "ttir.topk_router_gpt"(%input, %weight, %bias)
        <{k = 8 : i32, num_experts = 128 : i32}>
        : (tensor<32x64xbf16>, tensor<64x128xbf16>, tensor<32x128xbf16>)
          -> (tensor<32x8xui16>, tensor<32x8xbf16>)
    return %indices, %weights : tensor<32x8xui16>, tensor<32x8xbf16>
  }
}
