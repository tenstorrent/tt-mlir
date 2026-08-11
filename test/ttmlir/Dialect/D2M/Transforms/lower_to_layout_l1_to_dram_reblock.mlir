// RUN: ttmlir-opt --ttcore-register-device --d2m-lower-to-layout %s | FileCheck %s

#l1 = #ttcore.metal_layout<logical_shape = 576x4096, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#dram = #ttcore.metal_layout<logical_shape = 576x4096, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, dram, sharded>

// Reblock in L1 before transferring to a differently shaped DRAM grid. Copying
// directly from the 8x8 source grid would launch 64 writers for 48 DRAM shards.
// CHECK-LABEL: func.func @l1_to_dram_reblock
// CHECK: %[[DRAM_OUT:.*]] = d2m.empty() : tensor<6x8x3x16x!ttcore.tile<32x32, bf16>, #[[DRAM_LAYOUT:.*]]>
// CHECK: %[[SCALAR_8X8:.*]] = d2m.empty() : tensor<8x8x96x512xbf16, #[[L1_LAYOUT:.*]]>
// CHECK: %[[UNTILIZED:.*]] = d2m.generic
// CHECK-SAME: grid = #ttcore.grid<8x8>
// CHECK: ins(%arg0 : tensor<8x8x3x16x!ttcore.tile<32x32, bf16>, #[[L1_LAYOUT]]>)
// CHECK: outs(%[[SCALAR_8X8]] : tensor<8x8x96x512xbf16, #[[L1_LAYOUT]]>)
// CHECK: %[[SCALAR_6X8:.*]] = d2m.empty() : tensor<6x8x96x512xbf16, #[[L1_LAYOUT]]>
// CHECK: %[[SRC_VIEW:.*]] = d2m.view_layout %[[UNTILIZED]]
// CHECK: %[[REBLOCKED:.*]] = d2m.generic
// CHECK-SAME: grid = #ttcore.grid<6x8>
// CHECK: ins(%[[SRC_VIEW]] : tensor<6x8x96x512xbf16, #[[L1_LAYOUT]]>)
// CHECK: outs(%[[SCALAR_6X8]] : tensor<6x8x96x512xbf16, #[[L1_LAYOUT]]>)
// CHECK: %[[TILED_L1:.*]] = d2m.empty() : tensor<6x8x3x16x!ttcore.tile<32x32, bf16>, #[[L1_LAYOUT]]>
// CHECK: %[[TILIZED:.*]] = d2m.generic
// CHECK-SAME: grid = #ttcore.grid<6x8>
// CHECK: ins(%[[REBLOCKED]] : tensor<6x8x96x512xbf16, #[[L1_LAYOUT]]>)
// CHECK: outs(%[[TILED_L1]] : tensor<6x8x3x16x!ttcore.tile<32x32, bf16>, #[[L1_LAYOUT]]>)
// CHECK: %[[COPIED:.*]] = d2m.generic
// CHECK-SAME: grid = #ttcore.grid<6x8>
// CHECK: ins(%[[TILIZED]] : tensor<6x8x3x16x!ttcore.tile<32x32, bf16>, #[[L1_LAYOUT]]>)
// CHECK: outs(%[[DRAM_OUT]] : tensor<6x8x3x16x!ttcore.tile<32x32, bf16>, #[[DRAM_LAYOUT]]>)
// CHECK: return %[[COPIED]]
func.func @l1_to_dram_reblock(%arg0: tensor<8x8x3x16x!ttcore.tile<32x32, bf16>, #l1>) -> tensor<6x8x3x16x!ttcore.tile<32x32, bf16>, #dram> {
  %0 = d2m.empty() : tensor<6x8x3x16x!ttcore.tile<32x32, bf16>, #dram>
  %1 = d2m.to_layout %arg0, %0 : tensor<8x8x3x16x!ttcore.tile<32x32, bf16>, #l1> into tensor<6x8x3x16x!ttcore.tile<32x32, bf16>, #dram> -> tensor<6x8x3x16x!ttcore.tile<32x32, bf16>, #dram>
  return %1 : tensor<6x8x3x16x!ttcore.tile<32x32, bf16>, #dram>
}
