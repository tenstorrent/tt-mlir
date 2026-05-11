// RUN: ttmlir-opt --ttcore-register-device --d2m-lower-to-layout -o %t %s
// RUN: FileCheck %s --input-file=%t

#remap_layout = #ttcore.metal_layout<logical_shape = 256x256, dim_alignments = 32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, l1, sharded>
#oob_src = #ttcore.metal_layout<logical_shape = 1x1x32x33, dim_alignments = 1x1x32x32, collapsed_intervals = dense<[[0, 3], [3, 4]]> : tensor<2x2xi64>, l1, sharded>
#oob_dst = #ttcore.metal_layout<logical_shape = 1x1x32x33, dim_alignments = 1x1x32x32, collapsed_intervals = dense<[[0, 3], [3, 4]]> : tensor<2x2xi64>, l1, sharded>
#vgm_layout = #ttcore.metal_layout<logical_shape = 32x2048, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#remap_reblock_layout = #ttcore.metal_layout<logical_shape = 64x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#mask_retype_src = #ttcore.metal_layout<logical_shape = 60x60, dim_alignments = 128x128, collapsed_intervals = dense<[[0, 1]]> : tensor<1x2xi64>, l1, sharded>
#mask_retype_aligned = #ttcore.metal_layout<logical_shape = 128x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>

// CHECK-LABEL: func.func @remap_only_uses_view_layout
// CHECK: %[[RESULT:.*]] = d2m.view_layout %arg0 remapping = #map : tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #layout> -> tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #layout>
// CHECK-NOT: d2m.generic
// CHECK: return %[[RESULT]]
func.func @remap_only_uses_view_layout(%arg0: tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #remap_layout>) -> tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #remap_layout> {
  %0 = d2m.empty() : tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #remap_layout>
  %1 = d2m.view_layout %0 remapping = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)> : tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #remap_layout> -> tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #remap_layout>
  %2 = d2m.to_layout %arg0, %1 : tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #remap_layout> into tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #remap_layout> -> tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #remap_layout>
  return %2 : tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #remap_layout>
}

// CHECK-LABEL: func.func @remap_after_reblock_uses_backing_empty
// CHECK: %[[MATERIALIZED_EMPTY:.*]] = d2m.empty() : tensor<4x2x32x32xf32, #layout
// CHECK: %[[REBLOCK_VIEW:.*]] = d2m.view_layout %arg0 remapping
// CHECK: %[[REBLOCKED:.*]] = d2m.generic
// CHECK: ins(%[[REBLOCK_VIEW]] : tensor<4x2x32x32xf32, #layout
// CHECK: outs(%[[MATERIALIZED_EMPTY]] : tensor<4x2x32x32xf32, #layout
// CHECK: %[[RESULT:.*]] = d2m.view_layout %[[REBLOCKED]] remapping
// CHECK: return %[[RESULT]]
func.func @remap_after_reblock_uses_backing_empty(%arg0: tensor<2x4x32x32xf32, #remap_reblock_layout>) -> tensor<4x2x32x32xf32, #remap_reblock_layout> {
  %0 = d2m.empty() : tensor<4x2x32x32xf32, #remap_reblock_layout>
  %1 = d2m.view_layout %0 remapping = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)> : tensor<4x2x32x32xf32, #remap_reblock_layout> -> tensor<4x2x32x32xf32, #remap_reblock_layout>
  %2 = d2m.to_layout %arg0, %1 : tensor<2x4x32x32xf32, #remap_reblock_layout> into tensor<4x2x32x32xf32, #remap_reblock_layout> -> tensor<4x2x32x32xf32, #remap_reblock_layout>
  return %2 : tensor<4x2x32x32xf32, #remap_reblock_layout>
}

// CHECK-LABEL: func.func @oob_only_change_masks_explicitly
// CHECK: %[[DST:.*]] = d2m.empty() : tensor<1x1x1x2x!ttcore.tile<32x32, f32>, #layout{{[0-9]*}}>
// CHECK: d2m.mask %arg0, %[[DST]] logical_shape = [1, 1, 32, 33] fill_value = <zero>
// CHECK: return %{{.*}} : tensor<1x1x1x2x!ttcore.tile<32x32, f32>, #layout{{[0-9]*}}>
func.func @oob_only_change_masks_explicitly(%arg0: tensor<1x1x1x2x!ttcore.tile<32x32, f32>, #oob_src>) -> tensor<1x1x1x2x!ttcore.tile<32x32, f32>, #oob_dst> {
  %0 = d2m.empty() : tensor<1x1x1x2x!ttcore.tile<32x32, f32>, #oob_dst>
  %1 = d2m.to_layout %arg0, %0 : tensor<1x1x1x2x!ttcore.tile<32x32, f32>, #oob_src> into tensor<1x1x1x2x!ttcore.tile<32x32, f32>, #oob_dst> -> tensor<1x1x1x2x!ttcore.tile<32x32, f32>, #oob_dst>
  %2 = d2m.empty() : tensor<1x1x1x2x!ttcore.tile<32x32, f32>, #oob_dst>
  %3 = d2m.mask %1, %2 logical_shape = [1, 1, 32, 33] fill_value = <zero> : tensor<1x1x1x2x!ttcore.tile<32x32, f32>, #oob_dst> into tensor<1x1x1x2x!ttcore.tile<32x32, f32>, #oob_dst> -> tensor<1x1x1x2x!ttcore.tile<32x32, f32>, #oob_dst>
  return %3 : tensor<1x1x1x2x!ttcore.tile<32x32, f32>, #oob_dst>
}

// CHECK-LABEL: func.func @vgm_only_change_rebuffers_without_view
// CHECK: %[[SRC:.*]] = d2m.empty() {virtualGridForwardMapping = #map{{[0-9]*}}, virtualGridInverseMapping = #map{{[0-9]*}}} : tensor<1x64x1x1x!ttcore.tile<32x32, bf16>, #layout{{[0-9]*}}>
// CHECK: %[[DST:.*]] = d2m.empty() {virtualGridForwardMapping = #map{{[0-9]*}}, virtualGridInverseMapping = #map{{[0-9]*}}} : tensor<1x64x1x1x!ttcore.tile<32x32, bf16>, #layout{{[0-9]*}}>
// CHECK: %[[RESULT:.*]] = d2m.generic
// CHECK-SAME: threads = [#d2m.thread<unified>]
// CHECK: ins(%[[SRC]] : tensor<1x64x1x1x!ttcore.tile<32x32, bf16>, #layout{{[0-9]*}}>)
// CHECK: outs(%[[DST]] : tensor<1x64x1x1x!ttcore.tile<32x32, bf16>, #layout{{[0-9]*}}>)
// CHECK-NOT: d2m.view_layout
// CHECK: return %[[RESULT]]
func.func @vgm_only_change_rebuffers_without_view() -> tensor<1x64x1x1x!ttcore.tile<32x32, bf16>, #vgm_layout> {
  %0 = d2m.empty() {virtualGridForwardMapping = affine_map<(d0, d1, d2, d3) -> (d0 + 6, d1 + 7, d2, d3)>, virtualGridInverseMapping = affine_map<(d0, d1) -> (0, d0 - 6, d1 - 7)>} : tensor<1x64x1x1x!ttcore.tile<32x32, bf16>, #vgm_layout>
  %1 = d2m.empty() {virtualGridForwardMapping = affine_map<(d0, d1, d2, d3) -> (d0 + 4, d1 + 5, d2, d3)>, virtualGridInverseMapping = affine_map<(d0, d1) -> (0, d0 - 4, d1 - 5)>} : tensor<1x64x1x1x!ttcore.tile<32x32, bf16>, #vgm_layout>
  %2 = d2m.to_layout %0, %1 : tensor<1x64x1x1x!ttcore.tile<32x32, bf16>, #vgm_layout> into tensor<1x64x1x1x!ttcore.tile<32x32, bf16>, #vgm_layout> -> tensor<1x64x1x1x!ttcore.tile<32x32, bf16>, #vgm_layout>
  return %2 : tensor<1x64x1x1x!ttcore.tile<32x32, bf16>, #vgm_layout>
}

// CHECK-LABEL: func.func @mask_output_retyped_after_tilize
// CHECK: %[[TILIZED:.*]] = d2m.generic
// CHECK: %[[MASK_OUT:.*]] = d2m.empty() : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout{{[0-9]*}}>
// CHECK: %[[MASKED:.*]] = d2m.mask %[[TILIZED]], %[[MASK_OUT]] logical_shape = [60, 60] fill_value = <one> : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout{{[0-9]*}}> into tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout{{[0-9]*}}> -> tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout{{[0-9]*}}>
// CHECK: d2m.view_layout %[[MASKED]]
func.func @mask_output_retyped_after_tilize(%arg0: tensor<60x60xf32>) -> tensor<128x128xf32> {
  %0 = d2m.empty() : tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #mask_retype_src>
  %1 = d2m.to_layout %arg0, %0 : tensor<60x60xf32> into tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #mask_retype_src> -> tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #mask_retype_src>
  %2 = d2m.empty() : tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #mask_retype_src>
  %3 = d2m.mask %1, %2 logical_shape = [60, 60] fill_value = <one> : tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #mask_retype_src> into tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #mask_retype_src> -> tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #mask_retype_src>
  %4 = d2m.view_layout %3 remapping = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)> {reinterpretLayout = true} : tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #mask_retype_src> -> tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #mask_retype_aligned>
  %5 = d2m.empty() : tensor<128x128xf32>
  %6 = d2m.to_layout %4, %5 : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #mask_retype_aligned> into tensor<128x128xf32> -> tensor<128x128xf32>
  return %6 : tensor<128x128xf32>
}
