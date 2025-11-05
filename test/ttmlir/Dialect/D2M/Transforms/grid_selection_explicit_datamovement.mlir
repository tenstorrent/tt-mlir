// RUN: ttmlir-opt --split-input-file --ttcore-register-device --d2m-grid-selection %s | FileCheck %s

// Verify that d2m-grid-selection pass skips d2m.generic operations in
// explicit datamovement form. These operations have empty indexing_maps
// and users manage grids manually, so the pass should not attempt to
// assign or optimize grids.

#layout = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1>

// CHECK-LABEL: func.func @skip_grid_selection_explicit_datamovement
func.func @skip_grid_selection_explicit_datamovement(
  %arg0: tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>,
  %arg1: tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>
) -> tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout> {
  %0 = "d2m.empty"() : () -> tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>
  %stream = "d2m.stream_layout"(%arg0, %0) : (tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>, tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>) -> tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>

  // CHECK: d2m.generic
  // CHECK-SAME: grid = #ttcore.grid<1x1>
  // Grid should remain unchanged (1x1, not optimized to device grid)
  %result = "d2m.generic"(%stream, %arg1) <{
    block_factors = [],
    grid = #ttcore.grid<1x1>,
    indexing_maps = [],
    iterator_types = [],
    operandSegmentSizes = array<i32: 1, 1>,
    threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]
  }> ({
  ^bb0(%cb0: !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>>>, %cb1: !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>>>):
    %1 = "d2m.reserve"(%cb0) : (!d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>>>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  }, {
  ^bb0(%cb0: !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>>>, %cb1: !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>>>):
    %2 = "d2m.wait"(%cb0) : (!d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>>>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  }) : (tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>, tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>) -> tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>

  return %result : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>
}
