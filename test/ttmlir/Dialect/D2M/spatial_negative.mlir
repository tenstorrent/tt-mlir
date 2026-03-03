// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for d2m.spatial operation verification.

#layout = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>

// CHECK: error: 'd2m.spatial' op grid_ranges must contain at least one CoreRange

module {
  func.func @spatial_empty_grid_ranges(
      %arg0: tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>)
      -> tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout> {
    %0 = d2m.spatial {grid_ranges = #ttcore.core_range_set<>}
        ins() outs(%arg0 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>) {
      ^region_0:
        %1 = d2m.generic {
          block_factors = [1, 1],
          grid = #ttcore.grid<1x1>,
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = [#ttcore.iterator_type<parallel>,
                            #ttcore.iterator_type<parallel>],
          threads = [#d2m.thread<unified>]
        } ins() outs(%arg0 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>) {
        ^unified0(%cb_out: !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>>>):
          %out = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
          d2m.yield %out : (tensor<2x2x!ttcore.tile<32x32, f32>>)
        } : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>
        d2m.spatial_yield %1 : (tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>)
    } : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>
    return %0 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>
  }
}

// -----

#layout_2 = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>

// CHECK: error: 'd2m.spatial' op number of CoreRanges (2) must match the number of Regions (1)

module {
  func.func @spatial_core_ranges_mismatch_regions(
      %arg0: tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_2>)
      -> tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_2> {
    %0 = d2m.spatial {
      grid_ranges = #ttcore.core_range_set<[
        #ttcore.core_range<(0, 0), (0, 0)>,
        #ttcore.core_range<(1, 1), (1, 1)>
      ]>
    } ins() outs(%arg0 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_2>) {
      ^region_0:
        %1 = d2m.generic {
          block_factors = [1, 1],
          grid = #ttcore.grid<1x1>,
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = [#ttcore.iterator_type<parallel>,
                            #ttcore.iterator_type<parallel>],
          threads = [#d2m.thread<unified>]
        } ins() outs(%arg0 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_2>) {
        ^unified0(%cb_out: !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>>>):
          %out = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
          d2m.yield %out : (tensor<2x2x!ttcore.tile<32x32, f32>>)
        } : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_2>
        d2m.spatial_yield %1 : (tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_2>)
    } : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_2>
    return %0 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_2>
  }
}
