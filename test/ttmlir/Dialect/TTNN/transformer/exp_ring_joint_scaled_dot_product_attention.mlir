// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --split-input-file %s | FileCheck %s

// Ring SDPA is produced by a TTNN-level rewrite on already-partitioned IR, so
// there is no TTIR counterpart to lower from. These cases construct the op
// directly and check that it parses, verifies and round-trips.

// Q/K/V are sequence-sharded: [B=1, H=40, N=512, E=128] per device. With a ring
// size of 2 the persistent buffers hold the gathered sequence, N*2 = 1024.

// Self-attention, fully bound: what a finalized op looks like after both
// prelude passes have run.
module attributes {} {
  func.func @ring_sdpa_self_attention(
      %q: tensor<1x40x512x128xbf16>,
      %k: tensor<1x40x512x128xbf16>,
      %v: tensor<1x40x512x128xbf16>,
      %buf_k: tensor<1x40x1024x128xbf16>,
      %buf_v: tensor<1x40x1024x128xbf16>,
      %ping: !ttnn.global_semaphore,
      %pong: !ttnn.global_semaphore)
      -> tensor<1x40x512x128xbf16> {
    // CHECK-LABEL: @ring_sdpa_self_attention
    // CHECK: "ttnn.exp_ring_joint_scaled_dot_product_attention"
    // CHECK-SAME: cluster_axis = 0 : ui32
    // CHECK-SAME: dim = 2 : si32
    // CHECK-SAME: joint_strategy = "rear"
    // CHECK-SAME: logical_n = 1000 : i64
    // CHECK-SAME: operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 1, 1, 2>
    %0, %1, %2 = "ttnn.exp_ring_joint_scaled_dot_product_attention"(%q, %k, %v, %buf_k, %buf_v, %ping, %pong) <{
      operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 1, 1, 2>,
      joint_strategy = "rear",
      logical_n = 1000 : i64,
      dim = 2 : si32,
      cluster_axis = 0 : ui32,
      program_config = #ttnn.sdpa_program_config<
        compute_with_storage_grid_size = #ttnn.core_coord<8, 8>,
        q_chunk_size = 128,
        k_chunk_size = 128>
    }> : (tensor<1x40x512x128xbf16>, tensor<1x40x512x128xbf16>, tensor<1x40x512x128xbf16>, tensor<1x40x1024x128xbf16>, tensor<1x40x1024x128xbf16>, !ttnn.global_semaphore, !ttnn.global_semaphore) -> (tensor<1x40x512x128xbf16>, tensor<1x40x32x128xbf16>, tensor<1x40x512x32xf32>)
    return %0 : tensor<1x40x512x128xbf16>
  }
}

// -----

// Immediately after the rewrite: buffers and semaphores unbound, to be filled in
// by TTNNAllocateDistributedOpBuffers / ...Semaphores.
module attributes {} {
  func.func @ring_sdpa_unbound(
      %q: tensor<1x40x512x128xbf16>,
      %k: tensor<1x40x512x128xbf16>,
      %v: tensor<1x40x512x128xbf16>)
      -> tensor<1x40x512x128xbf16> {
    // CHECK-LABEL: @ring_sdpa_unbound
    // CHECK: "ttnn.exp_ring_joint_scaled_dot_product_attention"
    // CHECK-SAME: operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0, 0, 0>
    %0, %1, %2 = "ttnn.exp_ring_joint_scaled_dot_product_attention"(%q, %k, %v) <{
      operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0, 0, 0>,
      joint_strategy = "rear",
      logical_n = 1024 : i64,
      dim = 2 : si32,
      cluster_axis = 0 : ui32,
      program_config = #ttnn.sdpa_program_config<
        compute_with_storage_grid_size = #ttnn.core_coord<8, 8>,
        q_chunk_size = 128,
        k_chunk_size = 128>
    }> : (tensor<1x40x512x128xbf16>, tensor<1x40x512x128xbf16>, tensor<1x40x512x128xbf16>) -> (tensor<1x40x512x128xbf16>, tensor<1x40x32x128xbf16>, tensor<1x40x512x32xf32>)
    return %0 : tensor<1x40x512x128xbf16>
  }
}

// -----

// Joint (cross-attention) inputs present, plus the ring fabric tuning Wan uses
// and an explicit scale.
module attributes {} {
  func.func @ring_sdpa_joint(
      %q: tensor<1x40x512x128xbf16>,
      %k: tensor<1x40x512x128xbf16>,
      %v: tensor<1x40x512x128xbf16>,
      %jq: tensor<1x40x512x128xbf16>,
      %jk: tensor<1x40x512x128xbf16>,
      %jv: tensor<1x40x512x128xbf16>,
      %buf_k: tensor<1x40x4096x128xbf16>,
      %buf_v: tensor<1x40x4096x128xbf16>,
      %ping: !ttnn.global_semaphore,
      %pong: !ttnn.global_semaphore)
      -> tensor<1x40x512x128xbf16> {
    // CHECK-LABEL: @ring_sdpa_joint
    // CHECK: "ttnn.exp_ring_joint_scaled_dot_product_attention"
    // CHECK-SAME: num_buffers_per_channel = 32 : ui32
    // CHECK-SAME: num_links = 4 : ui32
    // CHECK-SAME: num_workers_per_link = 5 : ui32
    // CHECK-SAME: operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1, 1, 2>
    // CHECK-SAME: scale = {{.*}} : f32
    // CHECK-SAME: topology = #ttcore.topology<ring>
    %0, %1, %2 = "ttnn.exp_ring_joint_scaled_dot_product_attention"(%q, %k, %v, %jq, %jk, %jv, %buf_k, %buf_v, %ping, %pong) <{
      operandSegmentSizes = array<i32: 1, 1, 1, 1, 1, 1, 1, 1, 2>,
      joint_strategy = "rear",
      logical_n = 4096 : i64,
      dim = 2 : si32,
      cluster_axis = 1 : ui32,
      program_config = #ttnn.sdpa_program_config<
        compute_with_storage_grid_size = #ttnn.core_coord<8, 8>,
        q_chunk_size = 256,
        k_chunk_size = 256,
        exp_approx_mode = true>,
      num_links = 4 : ui32,
      topology = #ttcore.topology<ring>,
      scale = 8.83789062e-02 : f32,
      num_workers_per_link = 5 : ui32,
      num_buffers_per_channel = 32 : ui32
    }> : (tensor<1x40x512x128xbf16>, tensor<1x40x512x128xbf16>, tensor<1x40x512x128xbf16>, tensor<1x40x512x128xbf16>, tensor<1x40x512x128xbf16>, tensor<1x40x512x128xbf16>, tensor<1x40x4096x128xbf16>, tensor<1x40x4096x128xbf16>, !ttnn.global_semaphore, !ttnn.global_semaphore) -> (tensor<1x40x512x128xbf16>, tensor<1x40x512x128xbf16>, tensor<1x40x512x32xf32>)
    return %0 : tensor<1x40x512x128xbf16>
  }
}

// -----

// GQA: 40 query heads over 8 KV heads.
module attributes {} {
  func.func @ring_sdpa_gqa(
      %q: tensor<1x40x512x128xbf16>,
      %k: tensor<1x8x512x128xbf16>,
      %v: tensor<1x8x512x128xbf16>,
      %buf_k: tensor<1x8x1024x128xbf16>,
      %buf_v: tensor<1x8x1024x128xbf16>,
      %ping: !ttnn.global_semaphore,
      %pong: !ttnn.global_semaphore)
      -> tensor<1x40x512x128xbf16> {
    // CHECK-LABEL: @ring_sdpa_gqa
    // CHECK: "ttnn.exp_ring_joint_scaled_dot_product_attention"
    %0, %1, %2 = "ttnn.exp_ring_joint_scaled_dot_product_attention"(%q, %k, %v, %buf_k, %buf_v, %ping, %pong) <{
      operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 1, 1, 2>,
      joint_strategy = "rear",
      logical_n = 1024 : i64,
      dim = 2 : si32,
      cluster_axis = 0 : ui32,
      program_config = #ttnn.sdpa_program_config<
        compute_with_storage_grid_size = #ttnn.core_coord<8, 8>,
        q_chunk_size = 128,
        k_chunk_size = 128>
    }> : (tensor<1x40x512x128xbf16>, tensor<1x8x512x128xbf16>, tensor<1x8x512x128xbf16>, tensor<1x8x1024x128xbf16>, tensor<1x8x1024x128xbf16>, !ttnn.global_semaphore, !ttnn.global_semaphore) -> (tensor<1x40x512x128xbf16>, tensor<1x40x32x128xbf16>, tensor<1x40x512x32xf32>)
    return %0 : tensor<1x40x512x128xbf16>
  }
}
