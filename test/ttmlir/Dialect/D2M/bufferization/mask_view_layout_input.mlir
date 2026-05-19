// RUN: ttmlir-opt --ttcore-register-device --ttir-bufferization-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

// Test that MaskOp::bufferize correctly handles a view-encoded input by
// inserting a d2m.view_layout to reconcile the type mismatch between the mask
// input (view<N>-encoded, produced by a prior d2m.view_layout) and its output
// (shard<...>-encoded, produced by d2m.empty). Without the fix, the MaskOp
// verifier would reject the bufferized IR because input, output, and result
// types would not all be identical.

#l1 = #ttcore.memory_space<l1>
#layout = #ttcore.metal_layout<
  logical_shape = 64x64,
  dim_alignments = 32x32,
  collapsed_intervals = dense<[[0, -1]]> : tensor<1x2xi64>,
  l1, sharded
>
// Non-identity remapping so d2m.view_layout bufferizes to view<4> encoding.
#view_map = affine_map<(d0, d1, d2, d3) -> (d1, d0, d2, d3)>

// CHECK-LABEL: func.func @mask_with_view_layout_input
func.func @mask_with_view_layout_input() -> tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #layout> {
  // CHECK: %[[IN:.*]] = memref.alloc() : memref<2x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
  %input = d2m.empty() : tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #layout>

  // CHECK: %[[OUT:.*]] = memref.alloc() : memref<2x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
  %output = d2m.empty() : tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #layout>

  // d2m.view_layout with a non-identity remapping bufferizes to view<N>
  // encoding on the underlying shard-allocated buffer.
  // CHECK: %[[VIEW:.*]] = d2m.view_layout %[[IN]] remapping
  // CHECK-SAME: memref<2x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
  // CHECK-SAME: memref<2x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
  %view = d2m.view_layout %input remapping = #view_map
    : tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #layout>
    -> tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #layout>

  // MaskOp::bufferize must insert a d2m.view_layout to convert the view<4>-
  // encoded input to shard<...>-encoded so that input, output, and result
  // types are all identical, satisfying the MaskOp verifier.
  // CHECK: %[[FIX:.*]] = d2m.view_layout %[[VIEW]]
  // CHECK-SAME: memref<2x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
  // CHECK-SAME: memref<2x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
  // CHECK: d2m.mask %[[FIX]], %[[OUT]] logical_shape = [50, 64] fill_value = <zero>
  // CHECK-SAME: memref<2x2x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
  %result = d2m.mask %view, %output logical_shape = [50, 64] fill_value = <zero>
    : tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #layout>
    into tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #layout>
    -> tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #layout>

  return %result : tensor<2x2x1x1x!ttcore.tile<32x32, f32>, #layout>
}
