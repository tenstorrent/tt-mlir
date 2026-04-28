// RUN: ttmlir-opt --ttcore-register-device --d2m-lower-to-layout -o %t %s
// RUN: FileCheck %s --input-file=%t

#src = #ttcore.metal_layout<logical_shape = 32x5120, dim_alignments = 32x256, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>
#dst = #ttcore.metal_layout<logical_shape = 32x5120, dim_alignments = 32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, undef, l1, sharded>
#bfp_src = #ttcore.metal_layout<logical_shape = 256x256, dim_alignments = 256x256, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>
#bfp_dst = #ttcore.metal_layout<logical_shape = 256x256, dim_alignments = 128x128, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>

// CHECK-LABEL: func.func @preserve_virtual_grid_during_untilize
// CHECK: %[[SRC:.*]] = d2m.empty() {virtualGridForwardMapping = #map, virtualGridInverseMapping = #map1} : tensor<1x40x1x4x!ttcore.tile<32x32, f32>, #{{.*}}>
// CHECK: %[[SCALAR:.*]] = d2m.empty() {virtualGridForwardMapping = #map, virtualGridInverseMapping = #map1} : tensor<1x40x32x128xf32, #{{.*}}>
// CHECK: %[[UNTILIZED:.*]] = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x40
// CHECK: ins(%[[SRC]] : tensor<1x40x1x4x!ttcore.tile<32x32, f32>, #{{.*}}>)
// CHECK: outs(%[[SCALAR]] : tensor<1x40x32x128xf32, #{{.*}}>)
// CHECK: d2m.tile_untilize_block
// CHECK-NOT: tensor<1x8x32x640xf32
func.func @preserve_virtual_grid_during_untilize() -> tensor<1x1x1x160x!ttcore.tile<32x32, f32>, #dst> {
  %src_tensor = d2m.empty() {virtualGridForwardMapping = affine_map<(d0, d1, d2, d3) -> ((d1 floordiv 8) mod 5, d1 mod 8, d2, d3)>, virtualGridInverseMapping = affine_map<(d0, d1) -> (0, 0, (d1 + d0 * 8) mod 40)>} : tensor<1x40x1x4x!ttcore.tile<32x32, f32>, #src>
  %dst_tensor = d2m.empty() : tensor<1x1x1x160x!ttcore.tile<32x32, f32>, #dst>
  %result = d2m.to_layout %src_tensor, %dst_tensor : tensor<1x40x1x4x!ttcore.tile<32x32, f32>, #src> into tensor<1x1x1x160x!ttcore.tile<32x32, f32>, #dst> -> tensor<1x1x1x160x!ttcore.tile<32x32, f32>, #dst>
  return %result : tensor<1x1x1x160x!ttcore.tile<32x32, f32>, #dst>
}

// CHECK-LABEL: func.func @bfp_tiled_reblock_uses_bf16_scalar_bridge
// CHECK: %[[SRC:.*]] = d2m.empty() : tensor<8x8x1x1x!ttcore.tile<32x32, bfp_bf8>, #{{.*}}>
// CHECK: %[[SCALAR_SRC:.*]] = d2m.empty() : tensor<8x8x32x32xbf16, #{{.*}}>
// CHECK: d2m.generic
// CHECK: ins(%[[SRC]] : tensor<8x8x1x1x!ttcore.tile<32x32, bfp_bf8>, #{{.*}}>)
// CHECK: outs(%[[SCALAR_SRC]] : tensor<8x8x32x32xbf16, #{{.*}}>)
// CHECK: d2m.tile_untilize_block
// CHECK: %[[SCALAR_DST:.*]] = d2m.empty() : tensor<4x4x64x64xbf16, #{{.*}}>
// CHECK: d2m.view_layout %[[UNTILIZED:.*]] remapping
// CHECK-SAME: tensor<8x8x32x32xbf16
// CHECK-SAME: -> tensor<4x4x64x64xbf16
// CHECK: outs(%[[SCALAR_DST]] : tensor<4x4x64x64xbf16, #{{.*}}>)
// CHECK: d2m.tile_tilize_block
func.func @bfp_tiled_reblock_uses_bf16_scalar_bridge() -> tensor<4x4x2x2x!ttcore.tile<32x32, bfp_bf8>, #bfp_dst> {
  %src_tensor = d2m.empty() : tensor<8x8x1x1x!ttcore.tile<32x32, bfp_bf8>, #bfp_src>
  %dst_tensor = d2m.empty() : tensor<4x4x2x2x!ttcore.tile<32x32, bfp_bf8>, #bfp_dst>
  %result = d2m.to_layout %src_tensor, %dst_tensor : tensor<8x8x1x1x!ttcore.tile<32x32, bfp_bf8>, #bfp_src> into tensor<4x4x2x2x!ttcore.tile<32x32, bfp_bf8>, #bfp_dst> -> tensor<4x4x2x2x!ttcore.tile<32x32, bfp_bf8>, #bfp_dst>
  return %result : tensor<4x4x2x2x!ttcore.tile<32x32, bfp_bf8>, #bfp_dst>
}
