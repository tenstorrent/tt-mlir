// REQUIRES: opmodel
// RUN: ttmlir-opt --ttcore-register-device --ttcore-mark-functions-as-forward --ttnn-greedy-l1-spill-management %s -o %t.mlir
// RUN: FileCheck %s --input-file %t.mlir
//
// Test: forked producer with view-eligible reshape. Tensors are sized so
// that the L1 buffer is ~64 KiB per core (4096x512 bf16 distributed
// across 64 banks at 32 tiles per bank). With cap=0.14 the per-core
// budget is ~209 KiB. R1 = relu(P0) is the discriminator op: its
// overallPeakL1Usage is ~136 KiB (input + output buffers + CB), so:
//   alias-aware additionalL1 = 64 KiB, totalL1 = ~200 KiB ≤ 209 KiB → no spill.
//   double-count additionalL1 = 128 KiB, totalL1 = ~264 KiB > 209 KiB → spill.

#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>

// 4096x512 bf16 tile = 128*16 = 2048 tiles. On <1x1> interleaved L1 the
// memref is per-bank: 2048 tiles / 64 banks = 32 tiles per core = 64 KiB.
#ttnn_layout_dram_2d = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<128x16x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout_l1_2d = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<128x16x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
#ttnn_layout_l1_3d = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 4096 + d1, d2), <1x1>, memref<128x16x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>

module attributes {ttnn.tensor_l1_usage_cap = 1.400000e-01 : f32} {
  func.func @forked_view_no_spill(
      %arg0: tensor<4096x512xbf16, #ttnn_layout_dram_2d>,
      %arg1: tensor<4096x512xbf16, #ttnn_layout_dram_2d>,
      %arg2: tensor<4096x512xbf16, #ttnn_layout_dram_2d>,
      %arg3: tensor<4096x512xbf16, #ttnn_layout_dram_2d>)
      -> (tensor<4096x512xbf16, #ttnn_layout_l1_2d>, tensor<1x4096x512xbf16, #ttnn_layout_l1_3d>) {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device

    // P0: forked producer in L1. ~64 KiB per core.
    %P0 = "ttnn.add"(%arg0, %arg1) : (tensor<4096x512xbf16, #ttnn_layout_dram_2d>, tensor<4096x512xbf16, #ttnn_layout_dram_2d>) -> tensor<4096x512xbf16, #ttnn_layout_l1_2d>

    // V: zero-copy view of P0 (last dim unchanged, leading-1 reshape, tile-aligned).
    %V = "ttnn.reshape"(%P0) <{shape = [1 : i32, 4096 : i32, 512 : i32]}> : (tensor<4096x512xbf16, #ttnn_layout_l1_2d>) -> tensor<1x4096x512xbf16, #ttnn_layout_l1_3d>

    // T validates with both P0 and V live.
    // alias-aware: additionalL1 ≈ 64 KiB.
    // double-count: additionalL1 ≈ 128 KiB → spill (or eviction of P0 or V).
    %T = "ttnn.add"(%arg2, %arg3) : (tensor<4096x512xbf16, #ttnn_layout_dram_2d>, tensor<4096x512xbf16, #ttnn_layout_dram_2d>) -> tensor<4096x512xbf16, #ttnn_layout_l1_2d>

    // Single-input consumers keep P0 and T alive past T's validation
    // without inflating each consumer's overallPeakL1Usage to >budget.
    %R1 = "ttnn.relu"(%P0) : (tensor<4096x512xbf16, #ttnn_layout_l1_2d>) -> tensor<4096x512xbf16, #ttnn_layout_l1_2d>
    %R2 = "ttnn.relu"(%T) : (tensor<4096x512xbf16, #ttnn_layout_l1_2d>) -> tensor<4096x512xbf16, #ttnn_layout_l1_2d>

    return %R1, %V : tensor<4096x512xbf16, #ttnn_layout_l1_2d>, tensor<1x4096x512xbf16, #ttnn_layout_l1_3d>
  }

  // CHECK-LABEL: func.func @forked_view_no_spill
  // No DRAM demotions: alias-aware accounting keeps everything in L1.
  // CHECK-NOT: "ttnn.to_memory_config"{{.*}}#dram
}
