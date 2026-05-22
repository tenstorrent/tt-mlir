// RUN: ttmlir-opt --ttcore-register-device --d2m-lower-to-layout --d2m-materialize-view-returns %s | FileCheck %s

#layout_l1 = #ttcore.metal_layout<logical_shape = 1x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#layout_dram = #ttcore.metal_layout<logical_shape = 1x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, dram, sharded>

func.func @untilize_dram_tiled_to_l1_scalar(%arg0: tensor<1x4x1x1x!ttcore.tile<32x32, u32>, #layout_dram>) -> tensor<1x4x32x32xui32, #layout_l1> {
  // CHECK-LABEL: @untilize_dram_tiled_to_l1_scalar
  // CHECK: %[[SCALAR:.*]] = d2m.empty() : tensor<1x4x32x32xui32, #[[L1_LAYOUT:.*]]>
  // CHECK: %[[L1_TILED:.*]] = d2m.empty() : tensor<1x4x1x1x!ttcore.tile<32x32, u32>, #[[L1_LAYOUT]]>
  // CHECK-NOT: tensor<1x4x32x32x!ttcore.tile<32x32, u32>
  // CHECK: %[[DRAM_TO_L1:.*]] = d2m.generic
  // CHECK: ins(%arg0 : tensor<1x4x1x1x!ttcore.tile<32x32, u32>
  // CHECK: outs(%[[L1_TILED]] : tensor<1x4x1x1x!ttcore.tile<32x32, u32>, #[[L1_LAYOUT]]>)
  // CHECK: d2m.generic
  // CHECK: ins(%[[DRAM_TO_L1]] : tensor<1x4x1x1x!ttcore.tile<32x32, u32>, #[[L1_LAYOUT]]>)
  // CHECK: outs(%[[SCALAR]] : tensor<1x4x32x32xui32, #[[L1_LAYOUT]]>)
  // CHECK: d2m.tile_untilize_block
  %0 = d2m.empty() : tensor<1x4x32x32xui32, #layout_l1>
  %1 = d2m.to_layout %arg0, %0 : tensor<1x4x1x1x!ttcore.tile<32x32, u32>, #layout_dram> into tensor<1x4x32x32xui32, #layout_l1> -> tensor<1x4x32x32xui32, #layout_l1>
  return %1 : tensor<1x4x32x32xui32, #layout_l1>
}
