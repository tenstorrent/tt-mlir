// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s

// input_tensor must be 4D
module attributes {} {
  func.func @combine_input_rank(%input: tensor<128x2880xbf16>, %meta: tensor<1x2x128x4xi64>, %mapping: tensor<1x1x32x8xi64>) -> tensor<4x1x128x2880xbf16> {
    %0 = "ttnn.all_to_all_combine"(%input, %meta, %mapping) <{cluster_axis = 0 : i64, num_devices = 2 : i64, num_experts_per_tok = 4 : i64}> : (tensor<128x2880xbf16>, tensor<1x2x128x4xi64>, tensor<1x1x32x8xi64>) -> tensor<4x1x128x2880xbf16>
    return %0 : tensor<4x1x128x2880xbf16>
  }
}
// CHECK: error: 'ttnn.all_to_all_combine' op input_tensor must be a 4D tensor

// -----

// num_devices must be positive
module attributes {} {
  func.func @combine_num_devices(%input: tensor<4x2x128x2880xbf16>, %meta: tensor<1x2x128x4xi64>, %mapping: tensor<1x1x32x8xi64>) -> tensor<4x1x128x2880xbf16> {
    %0 = "ttnn.all_to_all_combine"(%input, %meta, %mapping) <{cluster_axis = 0 : i64, num_devices = 0 : i64, num_experts_per_tok = 4 : i64}> : (tensor<4x2x128x2880xbf16>, tensor<1x2x128x4xi64>, tensor<1x1x32x8xi64>) -> tensor<4x1x128x2880xbf16>
    return %0 : tensor<4x1x128x2880xbf16>
  }
}
// CHECK: error: 'ttnn.all_to_all_combine' op num_devices must be positive

// -----

// cluster_axis must be 0 or 1
module attributes {} {
  func.func @combine_cluster_axis(%input: tensor<4x2x128x2880xbf16>, %meta: tensor<1x2x128x4xi64>, %mapping: tensor<1x1x32x8xi64>) -> tensor<4x1x128x2880xbf16> {
    %0 = "ttnn.all_to_all_combine"(%input, %meta, %mapping) <{cluster_axis = 3 : i64, num_devices = 2 : i64, num_experts_per_tok = 4 : i64}> : (tensor<4x2x128x2880xbf16>, tensor<1x2x128x4xi64>, tensor<1x1x32x8xi64>) -> tensor<4x1x128x2880xbf16>
    return %0 : tensor<4x1x128x2880xbf16>
  }
}
// CHECK: error: 'ttnn.all_to_all_combine' op cluster_axis must be 0 or 1

// -----

// num_experts_per_tok must be positive
module attributes {} {
  func.func @combine_num_experts(%input: tensor<4x2x128x2880xbf16>, %meta: tensor<1x2x128x4xi64>, %mapping: tensor<1x1x32x8xi64>) -> tensor<4x1x128x2880xbf16> {
    %0 = "ttnn.all_to_all_combine"(%input, %meta, %mapping) <{cluster_axis = 0 : i64, num_devices = 2 : i64, num_experts_per_tok = 0 : i64}> : (tensor<4x2x128x2880xbf16>, tensor<1x2x128x4xi64>, tensor<1x1x32x8xi64>) -> tensor<4x1x128x2880xbf16>
    return %0 : tensor<4x1x128x2880xbf16>
  }
}
// CHECK: error: 'ttnn.all_to_all_combine' op num_experts_per_tok must be positive

// -----

// output_shard_dim must be 1 or 2
module attributes {} {
  func.func @combine_output_shard_dim(%input: tensor<4x2x128x2880xbf16>, %meta: tensor<1x2x128x4xi64>, %mapping: tensor<1x1x32x8xi64>) -> tensor<4x1x128x2880xbf16> {
    %0 = "ttnn.all_to_all_combine"(%input, %meta, %mapping) <{cluster_axis = 0 : i64, num_devices = 2 : i64, num_experts_per_tok = 4 : i64, output_shard_dim = 3 : i64}> : (tensor<4x2x128x2880xbf16>, tensor<1x2x128x4xi64>, tensor<1x1x32x8xi64>) -> tensor<4x1x128x2880xbf16>
    return %0 : tensor<4x1x128x2880xbf16>
  }
}
// CHECK: error: 'ttnn.all_to_all_combine' op output_shard_dim must be 1 or 2, got 3
