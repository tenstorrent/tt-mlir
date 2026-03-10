// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s

// expert_mapping must be 4D
module attributes {} {
  func.func @remap_mapping_rank(%topk: tensor<1x2x128x32xbf16>, %mapping: tensor<32x8xi64>, %meta: tensor<1x2x128x4xi64>) -> (tensor<1x2x128x4xbf16>, tensor<1x1x8x4xbf16>) {
    %0, %1 = "ttnn.moe_expert_token_remap"(%topk, %mapping, %meta) <{reduction_size = 32 : i64}> : (tensor<1x2x128x32xbf16>, tensor<32x8xi64>, tensor<1x2x128x4xi64>) -> (tensor<1x2x128x4xbf16>, tensor<1x1x8x4xbf16>)
    return %0, %1 : tensor<1x2x128x4xbf16>, tensor<1x1x8x4xbf16>
  }
}
// CHECK: error: 'ttnn.moe_expert_token_remap' op expert_mapping must be a 4D tensor

// -----

// expert_metadata must be 4D
module attributes {} {
  func.func @remap_metadata_rank(%topk: tensor<1x2x128x32xbf16>, %mapping: tensor<1x1x32x8xi64>, %meta: tensor<128x4xi64>) -> (tensor<1x2x128x4xbf16>, tensor<1x1x8x4xbf16>) {
    %0, %1 = "ttnn.moe_expert_token_remap"(%topk, %mapping, %meta) <{reduction_size = 32 : i64}> : (tensor<1x2x128x32xbf16>, tensor<1x1x32x8xi64>, tensor<128x4xi64>) -> (tensor<1x2x128x4xbf16>, tensor<1x1x8x4xbf16>)
    return %0, %1 : tensor<1x2x128x4xbf16>, tensor<1x1x8x4xbf16>
  }
}
// CHECK: error: 'ttnn.moe_expert_token_remap' op expert_metadata must be a 4D tensor
