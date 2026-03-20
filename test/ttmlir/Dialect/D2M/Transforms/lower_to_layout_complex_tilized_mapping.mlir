// RUN: ttmlir-opt --ttcore-register-device --d2m-lower-to-layout -o %t %s
// RUN: FileCheck %s --input-file=%t

#src = #ttcore.metal_layout<logical_shape = 32x5120, dim_alignments = 32x256, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>
#dst = #ttcore.metal_layout<logical_shape = 32x5120, dim_alignments = 32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, undef, l1, sharded>

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
