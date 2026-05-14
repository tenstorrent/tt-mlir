// RUN: ttmlir-opt --split-input-file --ttcore-register-device --d2m-grid-selection %s | FileCheck %s

// Test coverage for edge cases in GridAnalysis/GridSelection.

#layout = #ttcore.metal_layout<logical_shape = 256x256, dim_alignments = 32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, l1, sharded>
#parallel = #ttcore.iterator_type<parallel>

// Reinterpret view_layout operand — analysis classifies as None (reinterpret
// views are type casts whose grid must match the input, not reblocked).
// CHECK-LABEL: func.func @reinterpret_view_operand
func.func @reinterpret_view_operand(
  %arg0: tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #layout>
) -> tensor<256x256xf32> {
  %view = d2m.view_layout %arg0 remapping = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)> {reinterpretLayout = true} : tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #layout> -> tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #layout>
  %out = d2m.empty() : tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #layout>
  // CHECK: d2m.generic
  // CHECK-SAME: grid = #ttcore.grid<8x8>
  %result = d2m.generic {
    block_factors = [1, 1],
    grid = #ttcore.grid<1x1>,
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = [#parallel, #parallel],
    threads = [#d2m.thread<unified>]
  }
  ins(%view : tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #layout>)
  outs(%out : tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #layout>) {
  ^unified0:
    %tmp = tensor.empty() : tensor<8x8x!ttcore.tile<32x32, f32>>
    d2m.yield %tmp : (tensor<8x8x!ttcore.tile<32x32, f32>>)
  } : tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #layout>
  %host = d2m.empty() : tensor<256x256xf32>
  %back = d2m.to_layout %result, %host : tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #layout> into tensor<256x256xf32> -> tensor<256x256xf32>
  return %back : tensor<256x256xf32>
}

// -----

// d2m.skip_grid_selection attribute — generic should be left unchanged.
#layout2 = #ttcore.metal_layout<logical_shape = 256x256, dim_alignments = 32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, l1, sharded>
#parallel2 = #ttcore.iterator_type<parallel>

// CHECK-LABEL: func.func @skip_grid_selection_attr
func.func @skip_grid_selection_attr(
  %arg0: tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #layout2>
) -> tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #layout2> {
  %out = d2m.empty() : tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #layout2>
  // CHECK: d2m.generic
  // CHECK-SAME: grid = #ttcore.grid<1x1>
  %result = "d2m.generic"(%arg0, %out) <{
    block_factors = [1, 1],
    grid = #ttcore.grid<1x1>,
    indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = [#parallel2, #parallel2],
    operandSegmentSizes = array<i32: 1, 1, 0>,
    threads = [#d2m.thread<unified>]
  }> ({
  ^unified0:
    %tmp = tensor.empty() : tensor<8x8x!ttcore.tile<32x32, f32>>
    d2m.yield %tmp : (tensor<8x8x!ttcore.tile<32x32, f32>>)
  }) {d2m.skip_grid_selection} : (tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #layout2>, tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #layout2>) -> tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #layout2>
  return %result : tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #layout2>
}
