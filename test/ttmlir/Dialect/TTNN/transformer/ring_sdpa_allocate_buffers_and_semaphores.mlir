// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Straight after the rewrite: buffers and semaphores unbound.
// RUN: ttmlir-opt --ttcore-register-device="mesh-shape=1,2" --ttnn-fusing="enable-ring-sdpa=true" -o %t_fused.mlir %s
// RUN: FileCheck %s --check-prefix=FUSED --input-file=%t_fused.mlir
//
// After the buffer allocator: the two persistent K/V buffers are bound, the
// semaphore pool is still empty.
// RUN: ttmlir-opt --ttcore-register-device="mesh-shape=1,2" --ttnn-fusing="enable-ring-sdpa=true" --ttnn-allocate-distributed-op-buffers -o %t_buf.mlir %s
// RUN: FileCheck %s --check-prefix=BUFFERS --input-file=%t_buf.mlir
//
// After both: fully bound, which is what the runtime requires.
// RUN: ttmlir-opt --ttcore-register-device="mesh-shape=1,2" --ttnn-fusing="enable-ring-sdpa=true" --ttnn-allocate-distributed-op-buffers --ttnn-allocate-distributed-op-semaphores -o %t_all.mlir %s
// RUN: FileCheck %s --check-prefix=ALL --input-file=%t_all.mlir

// Exercises the two DistributedOpInterface prelude passes on the ring op. They
// walk every op implementing the interface, so no pass changes were needed --
// the op just has to implement allocateBuffers / allocateSemaphores.
//
// Buffers are bound before the optimizer (so their L1 footprint is budgeted)
// and semaphores after (their core range derives from the finalized shard
// spec), which is why the two run separately and in that order.

#dram = #ttnn.buffer_type<dram>
#sharded = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 128 + d2, d3), <1x1>, memref<32x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#gathered = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 2048 + d1 * 256 + d2, d3), <1x1>, memref<64x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module {
  func.func @ring_sdpa_prelude_binding(
      %q: tensor<1x8x128x64xbf16, #sharded>,
      %k: tensor<1x8x128x64xbf16, #sharded>,
      %v: tensor<1x8x128x64xbf16, #sharded>)
      -> tensor<1x8x128x64xbf16, #sharded> {

    // FUSED-LABEL: @ring_sdpa_prelude_binding
    // FUSED-NOT: "ttnn.empty"
    // FUSED-NOT: "ttnn.create_global_semaphore"
    // FUSED: "ttnn.exp_ring_joint_scaled_dot_product_attention"
    // FUSED-SAME: operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 0, 0, 0>

    // BUFFERS-LABEL: @ring_sdpa_prelude_binding
    // Two empties in the prelude, sized to the gathered sequence: 128 * 2 = 256.
    // BUFFERS: "ttnn.empty"
    // BUFFERS-SAME: shape = #ttnn.shape<1x8x256x64>
    // BUFFERS: "ttnn.empty"
    // BUFFERS-SAME: shape = #ttnn.shape<1x8x256x64>
    // BUFFERS-NOT: "ttnn.create_global_semaphore"
    // Buffers bound (slots 7 and 8), semaphore pool still empty.
    // BUFFERS: "ttnn.exp_ring_joint_scaled_dot_product_attention"
    // BUFFERS-SAME: operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 1, 1, 0>

    // ALL-LABEL: @ring_sdpa_prelude_binding
    // Both allocators insert immediately after ttnn.get_device, so the pass
    // that runs later (semaphores) ends up earlier in the block. Order between
    // the two groups is not meaningful; only that all four exist and are bound.
    // A two-deep ping-pong pool, matching tt-metal's ring all-gather.
    // ALL-DAG: "ttnn.create_global_semaphore"
    // ALL-DAG: "ttnn.create_global_semaphore"
    // ALL-DAG: "ttnn.empty"
    // ALL-DAG: "ttnn.empty"
    // ALL: "ttnn.exp_ring_joint_scaled_dot_product_attention"
    // ALL-SAME: operandSegmentSizes = array<i32: 1, 1, 1, 0, 0, 0, 1, 1, 2>

    %0 = "ttnn.all_gather"(%k) <{all_gather_dim = 2 : si32, cluster_axis = 1 : ui32}> : (tensor<1x8x128x64xbf16, #sharded>) -> tensor<1x8x256x64xbf16, #gathered>
    %1 = "ttnn.all_gather"(%v) <{all_gather_dim = 2 : si32, cluster_axis = 1 : ui32}> : (tensor<1x8x128x64xbf16, #sharded>) -> tensor<1x8x256x64xbf16, #gathered>
    %2 = "ttnn.scaled_dot_product_attention"(%q, %0, %1) <{is_causal = false, operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>}> : (tensor<1x8x128x64xbf16, #sharded>, tensor<1x8x256x64xbf16, #gathered>, tensor<1x8x256x64xbf16, #gathered>) -> tensor<1x8x128x64xbf16, #sharded>
    return %2 : tensor<1x8x128x64xbf16, #sharded>
  }
}
