// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttcore-register-device="mesh-shape=1,2" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

// Flatbuffer emission smoke test for
// `ttnn.exp_ring_joint_scaled_dot_product_attention`. HOST-SIDE ONLY: it
// registers a mock 1x2 device and confirms the op serializes without
// diagnostics. It is deliberately NOT under `test/ttmlir/Silicon/` because the
// ring kernel needs a real multi-chip mesh to execute, and the Silicon tree is
// scanned by `ttrt run` jobs that would try to run every flatbuffer they find.
//
// The persistent K/V buffers come from `ttnn.empty` and the semaphores from
// `ttnn.create_global_semaphore` rather than being block arguments, because
// serialization resolves both through the flatbuffer object cache -- they have
// to be produced by ops that were themselves serialized. This is also exactly
// the shape the phase-2 prelude passes will emit.

#dram = #ttnn.buffer_type<dram>

// Q/K/V: [1, 8, 128, 64] per device, sequence-sharded on dim 2.
#sharded_layout = #ttnn.ttnn_layout<
  (d0, d1, d2, d3) -> (d0 * 1024 + d1 * 128 + d2, d3),
  <1x1>,
  memref<32x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>
>

// Gathered K/V buffers: [1, 8, 256, 64] (ring size 2).
#gathered_layout = #ttnn.ttnn_layout<
  (d0, d1, d2, d3) -> (d0 * 2048 + d1 * 256 + d2, d3),
  <1x1>,
  memref<64x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>
>

// lse: [1, 8, 128, 32]
#lse_layout = #ttnn.ttnn_layout<
  (d0, d1, d2, d3) -> (d0 * 1024 + d1 * 128 + d2, d3),
  <1x1>,
  memref<32x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>
>

// CHECK: module attributes {ttcore.system_desc = #system_desc}
module {
  func.func @ring_sdpa_flatbuffer(
      %q: tensor<1x8x128x64xbf16, #sharded_layout>,
      %k: tensor<1x8x128x64xbf16, #sharded_layout>,
      %v: tensor<1x8x128x64xbf16, #sharded_layout>)
      -> tensor<1x8x128x64xbf16, #sharded_layout>
      attributes {tt.function_type = "forward_device"} {
    // CHECK-LABEL: @ring_sdpa_flatbuffer
    %device = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x2>}> : () -> !ttnn.device

    // CHECK: "ttnn.empty"
    %buf_k = "ttnn.empty"(%device) <{shape = #ttnn.shape<1x8x256x64>}> : (!ttnn.device) -> tensor<1x8x256x64xbf16, #gathered_layout>
    %buf_v = "ttnn.empty"(%device) <{shape = #ttnn.shape<1x8x256x64>}> : (!ttnn.device) -> tensor<1x8x256x64xbf16, #gathered_layout>

    // CHECK: "ttnn.create_global_semaphore"
    %ping = "ttnn.create_global_semaphore"(%device) <{core_range_set = #ttnn.core_range_set<[#ttnn.core_range<(0,0), (0,7)>]>, initial_value = 0 : ui32}> : (!ttnn.device) -> !ttnn.global_semaphore
    %pong = "ttnn.create_global_semaphore"(%device) <{core_range_set = #ttnn.core_range_set<[#ttnn.core_range<(0,0), (0,7)>]>, initial_value = 0 : ui32}> : (!ttnn.device) -> !ttnn.global_semaphore

    // CHECK: "ttnn.exp_ring_joint_scaled_dot_product_attention"
    // CHECK-SAME: logical_n = 200 : i64
    %0, %1, %2 = "ttnn.exp_ring_joint_scaled_dot_product_attention"(%q, %k, %v, %buf_k, %buf_v, %ping, %pong) <{
      operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 1, 1, 2>,
      joint_strategy = "rear",
      logical_n = 200 : i64,
      dim = 2 : si32,
      cluster_axis = 0 : ui32,
      program_config = #ttnn.sdpa_program_config<
        compute_with_storage_grid_size = #ttnn.core_coord<8, 8>,
        q_chunk_size = 128,
        k_chunk_size = 128,
        exp_approx_mode = true>,
      num_links = 1 : ui32,
      topology = #ttcore.topology<ring>,
      num_workers_per_link = 5 : ui32,
      num_buffers_per_channel = 32 : ui32
    }> : (tensor<1x8x128x64xbf16, #sharded_layout>, tensor<1x8x128x64xbf16, #sharded_layout>, tensor<1x8x128x64xbf16, #sharded_layout>, tensor<1x8x256x64xbf16, #gathered_layout>, tensor<1x8x256x64xbf16, #gathered_layout>, !ttnn.global_semaphore, !ttnn.global_semaphore) -> (tensor<1x8x128x64xbf16, #sharded_layout>, tensor<1x8x128x64xbf16, #sharded_layout>, tensor<1x8x128x32xf32, #lse_layout>)

    return %0 : tensor<1x8x128x64xbf16, #sharded_layout>
  }
}
