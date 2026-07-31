// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s

// dim must be the sequence axis (rank - 2)
module attributes {} {
  func.func @ring_sdpa_bad_dim(%q: tensor<1x40x512x128xbf16>, %k: tensor<1x40x512x128xbf16>, %v: tensor<1x40x512x128xbf16>) -> tensor<1x40x512x128xbf16> {
    %0, %1, %2 = "ttnn.exp_ring_joint_scaled_dot_product_attention"(%q, %k, %v) <{operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0, 0, 0>, joint_strategy = "rear", logical_n = 512 : i64, dim = 1 : si32, cluster_axis = 0 : ui32, program_config = #ttnn.sdpa_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, q_chunk_size = 128, k_chunk_size = 128>}> : (tensor<1x40x512x128xbf16>, tensor<1x40x512x128xbf16>, tensor<1x40x512x128xbf16>) -> (tensor<1x40x512x128xbf16>, tensor<1x40x32x128xbf16>, tensor<1x40x512x32xf32>)
    return %0 : tensor<1x40x512x128xbf16>
  }
}
// CHECK: error: 'ttnn.exp_ring_joint_scaled_dot_product_attention' op dim must be the sequence axis

// -----

// K/V must still be sequence-sharded: a gathered K/V means the all-gather was
// left in place and this op should never have been formed.
module attributes {} {
  func.func @ring_sdpa_gathered_kv(%q: tensor<1x40x512x128xbf16>, %k: tensor<1x40x1024x128xbf16>, %v: tensor<1x40x1024x128xbf16>) -> tensor<1x40x512x128xbf16> {
    %0, %1, %2 = "ttnn.exp_ring_joint_scaled_dot_product_attention"(%q, %k, %v) <{operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0, 0, 0>, joint_strategy = "rear", logical_n = 1024 : i64, dim = 2 : si32, cluster_axis = 0 : ui32, program_config = #ttnn.sdpa_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, q_chunk_size = 128, k_chunk_size = 128>}> : (tensor<1x40x512x128xbf16>, tensor<1x40x1024x128xbf16>, tensor<1x40x1024x128xbf16>) -> (tensor<1x40x512x128xbf16>, tensor<1x40x32x128xbf16>, tensor<1x40x512x32xf32>)
    return %0 : tensor<1x40x512x128xbf16>
  }
}
// CHECK: error: 'ttnn.exp_ring_joint_scaled_dot_product_attention' op Key/Value sequence length must match query sequence length

// -----

// joint_* are all-or-none
module attributes {} {
  func.func @ring_sdpa_partial_joint(%q: tensor<1x40x512x128xbf16>, %k: tensor<1x40x512x128xbf16>, %v: tensor<1x40x512x128xbf16>, %jq: tensor<1x40x512x128xbf16>) -> tensor<1x40x512x128xbf16> {
    %0, %1, %2 = "ttnn.exp_ring_joint_scaled_dot_product_attention"(%q, %k, %v, %jq) <{operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0, 0, 0, 0>, joint_strategy = "rear", logical_n = 512 : i64, dim = 2 : si32, cluster_axis = 0 : ui32, program_config = #ttnn.sdpa_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, q_chunk_size = 128, k_chunk_size = 128>}> : (tensor<1x40x512x128xbf16>, tensor<1x40x512x128xbf16>, tensor<1x40x512x128xbf16>, tensor<1x40x512x128xbf16>) -> (tensor<1x40x512x128xbf16>, tensor<1x40x32x128xbf16>, tensor<1x40x512x32xf32>)
    return %0 : tensor<1x40x512x128xbf16>
  }
}
// CHECK: error: 'ttnn.exp_ring_joint_scaled_dot_product_attention' op joint_query, joint_key and joint_value must all be present or all be absent

// -----

// the persistent buffers are all-or-none
module attributes {} {
  func.func @ring_sdpa_partial_buffers(%q: tensor<1x40x512x128xbf16>, %k: tensor<1x40x512x128xbf16>, %v: tensor<1x40x512x128xbf16>, %buf_k: tensor<1x40x1024x128xbf16>) -> tensor<1x40x512x128xbf16> {
    %0, %1, %2 = "ttnn.exp_ring_joint_scaled_dot_product_attention"(%q, %k, %v, %buf_k) <{operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 1, 0, 0>, joint_strategy = "rear", logical_n = 1024 : i64, dim = 2 : si32, cluster_axis = 0 : ui32, program_config = #ttnn.sdpa_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, q_chunk_size = 128, k_chunk_size = 128>}> : (tensor<1x40x512x128xbf16>, tensor<1x40x512x128xbf16>, tensor<1x40x512x128xbf16>, tensor<1x40x1024x128xbf16>) -> (tensor<1x40x512x128xbf16>, tensor<1x40x32x128xbf16>, tensor<1x40x512x32xf32>)
    return %0 : tensor<1x40x512x128xbf16>
  }
}
// CHECK: error: 'ttnn.exp_ring_joint_scaled_dot_product_attention' op persistent_output_buffer_k and persistent_output_buffer_v must both be present or both be absent

// -----

// a single semaphore is not a ping-pong pool
module attributes {} {
  func.func @ring_sdpa_one_semaphore(%q: tensor<1x40x512x128xbf16>, %k: tensor<1x40x512x128xbf16>, %v: tensor<1x40x512x128xbf16>, %buf_k: tensor<1x40x1024x128xbf16>, %buf_v: tensor<1x40x1024x128xbf16>, %ping: !ttnn.global_semaphore) -> tensor<1x40x512x128xbf16> {
    %0, %1, %2 = "ttnn.exp_ring_joint_scaled_dot_product_attention"(%q, %k, %v, %buf_k, %buf_v, %ping) <{operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 1, 1, 1>, joint_strategy = "rear", logical_n = 1024 : i64, dim = 2 : si32, cluster_axis = 0 : ui32, program_config = #ttnn.sdpa_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, q_chunk_size = 128, k_chunk_size = 128>}> : (tensor<1x40x512x128xbf16>, tensor<1x40x512x128xbf16>, tensor<1x40x512x128xbf16>, tensor<1x40x1024x128xbf16>, tensor<1x40x1024x128xbf16>, !ttnn.global_semaphore) -> (tensor<1x40x512x128xbf16>, tensor<1x40x32x128xbf16>, tensor<1x40x512x32xf32>)
    return %0 : tensor<1x40x512x128xbf16>
  }
}
// CHECK: error: 'ttnn.exp_ring_joint_scaled_dot_product_attention' op multi_device_global_semaphore must be empty (before prelude allocation) or hold at least 2 semaphores

// -----

// semaphores bound before buffers is out of pass order: buffers are allocated
// before the optimizer, semaphores after.
module attributes {} {
  func.func @ring_sdpa_semaphores_before_buffers(%q: tensor<1x40x512x128xbf16>, %k: tensor<1x40x512x128xbf16>, %v: tensor<1x40x512x128xbf16>, %ping: !ttnn.global_semaphore, %pong: !ttnn.global_semaphore) -> tensor<1x40x512x128xbf16> {
    %0, %1, %2 = "ttnn.exp_ring_joint_scaled_dot_product_attention"(%q, %k, %v, %ping, %pong) <{operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0, 0, 2>, joint_strategy = "rear", logical_n = 1024 : i64, dim = 2 : si32, cluster_axis = 0 : ui32, program_config = #ttnn.sdpa_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, q_chunk_size = 128, k_chunk_size = 128>}> : (tensor<1x40x512x128xbf16>, tensor<1x40x512x128xbf16>, tensor<1x40x512x128xbf16>, !ttnn.global_semaphore, !ttnn.global_semaphore) -> (tensor<1x40x512x128xbf16>, tensor<1x40x32x128xbf16>, tensor<1x40x512x32xf32>)
    return %0 : tensor<1x40x512x128xbf16>
  }
}
// CHECK: error: 'ttnn.exp_ring_joint_scaled_dot_product_attention' op semaphores are bound but the persistent buffers are not

// -----

// a buffer sequence length that is not a whole ring's worth of K
module attributes {} {
  func.func @ring_sdpa_bad_buffer_seq(%q: tensor<1x40x512x128xbf16>, %k: tensor<1x40x512x128xbf16>, %v: tensor<1x40x512x128xbf16>, %buf_k: tensor<1x40x900x128xbf16>, %buf_v: tensor<1x40x900x128xbf16>, %ping: !ttnn.global_semaphore, %pong: !ttnn.global_semaphore) -> tensor<1x40x512x128xbf16> {
    %0, %1, %2 = "ttnn.exp_ring_joint_scaled_dot_product_attention"(%q, %k, %v, %buf_k, %buf_v, %ping, %pong) <{operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 1, 1, 2>, joint_strategy = "rear", logical_n = 900 : i64, dim = 2 : si32, cluster_axis = 0 : ui32, program_config = #ttnn.sdpa_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, q_chunk_size = 128, k_chunk_size = 128>}> : (tensor<1x40x512x128xbf16>, tensor<1x40x512x128xbf16>, tensor<1x40x512x128xbf16>, tensor<1x40x900x128xbf16>, tensor<1x40x900x128xbf16>, !ttnn.global_semaphore, !ttnn.global_semaphore) -> (tensor<1x40x512x128xbf16>, tensor<1x40x32x128xbf16>, tensor<1x40x512x32xf32>)
    return %0 : tensor<1x40x512x128xbf16>
  }
}
// CHECK: error: 'ttnn.exp_ring_joint_scaled_dot_product_attention' op persistent buffer sequence length

// -----

// a buffer that disagrees with K on a non-sequence dim
module attributes {} {
  func.func @ring_sdpa_bad_buffer_heads(%q: tensor<1x40x512x128xbf16>, %k: tensor<1x40x512x128xbf16>, %v: tensor<1x40x512x128xbf16>, %buf_k: tensor<1x20x1024x128xbf16>, %buf_v: tensor<1x20x1024x128xbf16>, %ping: !ttnn.global_semaphore, %pong: !ttnn.global_semaphore) -> tensor<1x40x512x128xbf16> {
    %0, %1, %2 = "ttnn.exp_ring_joint_scaled_dot_product_attention"(%q, %k, %v, %buf_k, %buf_v, %ping, %pong) <{operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 1, 1, 2>, joint_strategy = "rear", logical_n = 1024 : i64, dim = 2 : si32, cluster_axis = 0 : ui32, program_config = #ttnn.sdpa_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, q_chunk_size = 128, k_chunk_size = 128>}> : (tensor<1x40x512x128xbf16>, tensor<1x40x512x128xbf16>, tensor<1x40x512x128xbf16>, tensor<1x20x1024x128xbf16>, tensor<1x20x1024x128xbf16>, !ttnn.global_semaphore, !ttnn.global_semaphore) -> (tensor<1x40x512x128xbf16>, tensor<1x40x32x128xbf16>, tensor<1x40x512x32xf32>)
    return %0 : tensor<1x40x512x128xbf16>
  }
}
// CHECK: error: 'ttnn.exp_ring_joint_scaled_dot_product_attention' op persistent buffer dim 1 must match key dim 1

// -----

// logical_n cannot exceed the gathered sequence length
module attributes {} {
  func.func @ring_sdpa_logical_n_too_large(%q: tensor<1x40x512x128xbf16>, %k: tensor<1x40x512x128xbf16>, %v: tensor<1x40x512x128xbf16>, %buf_k: tensor<1x40x1024x128xbf16>, %buf_v: tensor<1x40x1024x128xbf16>, %ping: !ttnn.global_semaphore, %pong: !ttnn.global_semaphore) -> tensor<1x40x512x128xbf16> {
    %0, %1, %2 = "ttnn.exp_ring_joint_scaled_dot_product_attention"(%q, %k, %v, %buf_k, %buf_v, %ping, %pong) <{operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 1, 1, 2>, joint_strategy = "rear", logical_n = 2048 : i64, dim = 2 : si32, cluster_axis = 0 : ui32, program_config = #ttnn.sdpa_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, q_chunk_size = 128, k_chunk_size = 128>}> : (tensor<1x40x512x128xbf16>, tensor<1x40x512x128xbf16>, tensor<1x40x512x128xbf16>, tensor<1x40x1024x128xbf16>, tensor<1x40x1024x128xbf16>, !ttnn.global_semaphore, !ttnn.global_semaphore) -> (tensor<1x40x512x128xbf16>, tensor<1x40x32x128xbf16>, tensor<1x40x512x32xf32>)
    return %0 : tensor<1x40x512x128xbf16>
  }
}
// CHECK: error: 'ttnn.exp_ring_joint_scaled_dot_product_attention' op logical_n

// -----

// query num heads must be divisible by kv num heads
module attributes {} {
  func.func @ring_sdpa_bad_head_ratio(%q: tensor<1x40x512x128xbf16>, %k: tensor<1x12x512x128xbf16>, %v: tensor<1x12x512x128xbf16>) -> tensor<1x40x512x128xbf16> {
    %0, %1, %2 = "ttnn.exp_ring_joint_scaled_dot_product_attention"(%q, %k, %v) <{operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0, 0, 0>, joint_strategy = "rear", logical_n = 512 : i64, dim = 2 : si32, cluster_axis = 0 : ui32, program_config = #ttnn.sdpa_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, q_chunk_size = 128, k_chunk_size = 128>}> : (tensor<1x40x512x128xbf16>, tensor<1x12x512x128xbf16>, tensor<1x12x512x128xbf16>) -> (tensor<1x40x512x128xbf16>, tensor<1x40x32x128xbf16>, tensor<1x40x512x32xf32>)
    return %0 : tensor<1x40x512x128xbf16>
  }
}
// CHECK: error: 'ttnn.exp_ring_joint_scaled_dot_product_attention' op Query num heads must be divisible by key/value num heads

// -----

// query must be 4D
module attributes {} {
  func.func @ring_sdpa_bad_rank(%q: tensor<40x512x128xbf16>, %k: tensor<40x512x128xbf16>, %v: tensor<40x512x128xbf16>) -> tensor<40x512x128xbf16> {
    %0, %1, %2 = "ttnn.exp_ring_joint_scaled_dot_product_attention"(%q, %k, %v) <{operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0, 0, 0>, joint_strategy = "rear", logical_n = 512 : i64, dim = 1 : si32, cluster_axis = 0 : ui32, program_config = #ttnn.sdpa_program_config<compute_with_storage_grid_size = #ttnn.core_coord<8, 8>, q_chunk_size = 128, k_chunk_size = 128>}> : (tensor<40x512x128xbf16>, tensor<40x512x128xbf16>, tensor<40x512x128xbf16>) -> (tensor<40x512x128xbf16>, tensor<40x32x128xbf16>, tensor<40x512x32xf32>)
    return %0 : tensor<40x512x128xbf16>
  }
}
// CHECK: error: 'ttnn.exp_ring_joint_scaled_dot_product_attention' op Query must be a 4D tensor
