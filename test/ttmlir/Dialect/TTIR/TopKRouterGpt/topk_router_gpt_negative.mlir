// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --split-input-file --verify-diagnostics %s

// Bad k: k must be positive.
module {
  func.func @topk_router_gpt_bad_k(
      %input: tensor<32x64xbf16>, %weight: tensor<64x128xbf16>, %bias: tensor<32x128xbf16>
  ) -> (tensor<32x4xui16>, tensor<32x4xbf16>) {
    // expected-error @+1 {{'ttir.topk_router_gpt' op k must be positive, but got: 0}}
    %0, %1 = "ttir.topk_router_gpt"(%input, %weight, %bias)
        <{k = 0 : i32, num_experts = 128 : i32}>
        : (tensor<32x64xbf16>, tensor<64x128xbf16>, tensor<32x128xbf16>)
          -> (tensor<32x4xui16>, tensor<32x4xbf16>)
    return %0, %1 : tensor<32x4xui16>, tensor<32x4xbf16>
  }
}

// -----

// Wrong input rank: input must be 2D.
module {
  func.func @topk_router_gpt_bad_input_rank(
      %input: tensor<1x32x64xbf16>, %weight: tensor<64x128xbf16>, %bias: tensor<32x128xbf16>
  ) -> (tensor<32x4xui16>, tensor<32x4xbf16>) {
    // expected-error @+1 {{'ttir.topk_router_gpt' op input must be a 2D tensor [B, hidden_dim], but got rank 3}}
    %0, %1 = "ttir.topk_router_gpt"(%input, %weight, %bias)
        <{k = 4 : i32, num_experts = 128 : i32}>
        : (tensor<1x32x64xbf16>, tensor<64x128xbf16>, tensor<32x128xbf16>)
          -> (tensor<32x4xui16>, tensor<32x4xbf16>)
    return %0, %1 : tensor<32x4xui16>, tensor<32x4xbf16>
  }
}

// -----

// Dimension mismatch: weight dim 0 != input hidden_dim.
module {
  func.func @topk_router_gpt_weight_dim_mismatch(
      %input: tensor<32x64xbf16>, %weight: tensor<32x128xbf16>, %bias: tensor<32x128xbf16>
  ) -> (tensor<32x4xui16>, tensor<32x4xbf16>) {
    // expected-error @+1 {{'ttir.topk_router_gpt' op weight dim 0 (32) must equal input hidden_dim (64)}}
    %0, %1 = "ttir.topk_router_gpt"(%input, %weight, %bias)
        <{k = 4 : i32, num_experts = 128 : i32}>
        : (tensor<32x64xbf16>, tensor<32x128xbf16>, tensor<32x128xbf16>)
          -> (tensor<32x4xui16>, tensor<32x4xbf16>)
    return %0, %1 : tensor<32x4xui16>, tensor<32x4xbf16>
  }
}

// -----

// num_experts attribute mismatch: num_experts != weight.shape[1].
module {
  func.func @topk_router_gpt_num_experts_mismatch(
      %input: tensor<32x64xbf16>, %weight: tensor<64x128xbf16>, %bias: tensor<32x128xbf16>
  ) -> (tensor<32x4xui16>, tensor<32x4xbf16>) {
    // expected-error @+1 {{'ttir.topk_router_gpt' op num_experts attribute (64) must equal weight dim 1 (128)}}
    %0, %1 = "ttir.topk_router_gpt"(%input, %weight, %bias)
        <{k = 4 : i32, num_experts = 64 : i32}>
        : (tensor<32x64xbf16>, tensor<64x128xbf16>, tensor<32x128xbf16>)
          -> (tensor<32x4xui16>, tensor<32x4xbf16>)
    return %0, %1 : tensor<32x4xui16>, tensor<32x4xbf16>
  }
}

// -----

// Output shape mismatch: expert_indices dim 1 != k.
module {
  func.func @topk_router_gpt_bad_output_k(
      %input: tensor<32x64xbf16>, %weight: tensor<64x128xbf16>, %bias: tensor<32x128xbf16>
  ) -> (tensor<32x8xui16>, tensor<32x4xbf16>) {
    // expected-error @+1 {{'ttir.topk_router_gpt' op expert_indices dim 1 (8) must equal k (4)}}
    %0, %1 = "ttir.topk_router_gpt"(%input, %weight, %bias)
        <{k = 4 : i32, num_experts = 128 : i32}>
        : (tensor<32x64xbf16>, tensor<64x128xbf16>, tensor<32x128xbf16>)
          -> (tensor<32x8xui16>, tensor<32x4xbf16>)
    return %0, %1 : tensor<32x8xui16>, tensor<32x4xbf16>
  }
}

// -----

// Output batch mismatch: expert_indices dim 0 != B.
module {
  func.func @topk_router_gpt_bad_output_batch(
      %input: tensor<32x64xbf16>, %weight: tensor<64x128xbf16>, %bias: tensor<32x128xbf16>
  ) -> (tensor<16x4xui16>, tensor<32x4xbf16>) {
    // expected-error @+1 {{'ttir.topk_router_gpt' op expert_indices dim 0 (16) must equal input batch size B (32)}}
    %0, %1 = "ttir.topk_router_gpt"(%input, %weight, %bias)
        <{k = 4 : i32, num_experts = 128 : i32}>
        : (tensor<32x64xbf16>, tensor<64x128xbf16>, tensor<32x128xbf16>)
          -> (tensor<16x4xui16>, tensor<32x4xbf16>)
    return %0, %1 : tensor<16x4xui16>, tensor<32x4xbf16>
  }
}
