// RUN: ttmlir-opt --canonicalize %s | FileCheck %s

#layout_aligned = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, l1, sharded>
#layout_unaligned = #ttcore.metal_layout<logical_shape = 50x50, dim_alignments = 32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, l1, sharded>

// CHECK-LABEL: func.func @fold_undef_mask
func.func @fold_undef_mask(%arg0: tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_unaligned>,
                           %arg1: tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_unaligned>)
    -> tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_unaligned> {
  // CHECK-NOT: d2m.mask
  // CHECK: return %arg0
  %0 = d2m.mask %arg0, %arg1 logical_shape = [50, 50] fill_value = <undef> : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_unaligned> into tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_unaligned> -> tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_unaligned>
  return %0 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_unaligned>
}

// CHECK-LABEL: func.func @fold_aligned_mask
func.func @fold_aligned_mask(%arg0: tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_aligned>,
                             %arg1: tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_aligned>)
    -> tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_aligned> {
  // CHECK-NOT: d2m.mask
  // CHECK: return %arg0
  %0 = d2m.mask %arg0, %arg1 logical_shape = [64, 64] fill_value = <zero> : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_aligned> into tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_aligned> -> tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_aligned>
  return %0 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_aligned>
}

// CHECK-LABEL: func.func @keep_unaligned_mask
func.func @keep_unaligned_mask(%arg0: tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_unaligned>,
                               %arg1: tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_unaligned>)
    -> tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_unaligned> {
  // CHECK: d2m.mask
  // CHECK-SAME: logical_shape = [50, 50]
  // CHECK-SAME: fill_value = <zero>
  %0 = d2m.mask %arg0, %arg1 logical_shape = [50, 50] fill_value = <zero> : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_unaligned> into tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_unaligned> -> tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_unaligned>
  return %0 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_unaligned>
}
