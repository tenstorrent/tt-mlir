// RUN: ttmlir-opt --ttcore-register-device --d2m-lower-to-layout -o %t %s
// RUN: FileCheck %s --input-file=%t

#remap_layout = #ttcore.metal_layout<logical_shape = 256x256, dim_alignments = 32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, undef, l1, sharded>
#oob_src = #ttcore.metal_layout<logical_shape = 1x1x32x33, dim_alignments = 1x1x32x32, collapsed_intervals = dense<[[0, 3], [3, 4]]> : tensor<2x2xi64>, undef, l1, sharded>
#oob_dst = #ttcore.metal_layout<logical_shape = 1x1x32x33, dim_alignments = 1x1x32x32, collapsed_intervals = dense<[[0, 3], [3, 4]]> : tensor<2x2xi64>, zero, l1, sharded>
#vgm_layout = #ttcore.metal_layout<logical_shape = 32x2048, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>

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

// CHECK-LABEL: func.func @oob_only_change_reinterprets_then_masks
// CHECK: %[[REINT:.*]] = d2m.view_layout %arg0 remapping = #map1 {reinterpretLayout = true} : tensor<1x1x1x2x!ttcore.tile<32x32, f32>, #layout1> -> tensor<1x1x1x2x!ttcore.tile<32x32, f32>, #layout2>
// CHECK: %[[DST:.*]] = d2m.empty() : tensor<1x1x1x2x!ttcore.tile<32x32, f32>, #layout2>
// CHECK: d2m.generic
// CHECK: ins(%[[REINT]]
// CHECK: outs(%{{.*}} : tensor<1x1x1x2x!ttcore.tile<32x32, f32>, #layout2>)
// CHECK: d2m.block_mask
// CHECK-SAME: <zero>
// CHECK: return %{{.*}} : tensor<1x1x1x2x!ttcore.tile<32x32, f32>, #layout2>
func.func @oob_only_change_reinterprets_then_masks(%arg0: tensor<1x1x1x2x!ttcore.tile<32x32, f32>, #oob_src>) -> tensor<1x1x1x2x!ttcore.tile<32x32, f32>, #oob_dst> {
  %0 = d2m.empty() : tensor<1x1x1x2x!ttcore.tile<32x32, f32>, #oob_dst>
  %1 = d2m.to_layout %arg0, %0 : tensor<1x1x1x2x!ttcore.tile<32x32, f32>, #oob_src> into tensor<1x1x1x2x!ttcore.tile<32x32, f32>, #oob_dst> -> tensor<1x1x1x2x!ttcore.tile<32x32, f32>, #oob_dst>
  return %1 : tensor<1x1x1x2x!ttcore.tile<32x32, f32>, #oob_dst>
}

// CHECK-LABEL: func.func @vgm_only_change_rebuffers_without_view
// CHECK: %[[SRC:.*]] = d2m.empty() {virtualGridForwardMapping = #map4, virtualGridInverseMapping = #map5} : tensor<1x64x1x1x!ttcore.tile<32x32, bf16>, #layout4>
// CHECK: %[[DST:.*]] = d2m.empty() {virtualGridForwardMapping = #map6, virtualGridInverseMapping = #map7} : tensor<1x64x1x1x!ttcore.tile<32x32, bf16>, #layout4>
// CHECK: %[[RESULT:.*]] = d2m.generic
// CHECK-SAME: threads = [#d2m.thread<unified>]
// CHECK: ins(%[[SRC]] : tensor<1x64x1x1x!ttcore.tile<32x32, bf16>, #layout4>)
// CHECK: outs(%[[DST]] : tensor<1x64x1x1x!ttcore.tile<32x32, bf16>, #layout4>)
// CHECK-NOT: d2m.view_layout
// CHECK: return %[[RESULT]]
func.func @vgm_only_change_rebuffers_without_view() -> tensor<1x64x1x1x!ttcore.tile<32x32, bf16>, #vgm_layout> {
  %0 = d2m.empty() {virtualGridForwardMapping = affine_map<(d0, d1, d2, d3) -> (d0 + 6, d1 + 7, d2, d3)>, virtualGridInverseMapping = affine_map<(d0, d1) -> (0, d0 - 6, d1 - 7)>} : tensor<1x64x1x1x!ttcore.tile<32x32, bf16>, #vgm_layout>
  %1 = d2m.empty() {virtualGridForwardMapping = affine_map<(d0, d1, d2, d3) -> (d0 + 4, d1 + 5, d2, d3)>, virtualGridInverseMapping = affine_map<(d0, d1) -> (0, d0 - 4, d1 - 5)>} : tensor<1x64x1x1x!ttcore.tile<32x32, bf16>, #vgm_layout>
  %2 = d2m.to_layout %0, %1 : tensor<1x64x1x1x!ttcore.tile<32x32, bf16>, #vgm_layout> into tensor<1x64x1x1x!ttcore.tile<32x32, bf16>, #vgm_layout> -> tensor<1x64x1x1x!ttcore.tile<32x32, bf16>, #vgm_layout>
  return %2 : tensor<1x64x1x1x!ttcore.tile<32x32, bf16>, #vgm_layout>
}
