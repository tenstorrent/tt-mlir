// RUN: ttmlir-opt --split-input-file --ttcore-register-device --d2m-grid-selection %s | FileCheck %s

// Verify that d2m-grid-selection pass preserves d2m.spatial and runs
// reconstructSpatialOperands. The generic inside spatial uses the region's
// grid range; the pass should not change the spatial structure.

#any_device = #ttcore.device<workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1] -> (0, 0, 0, d0 * s1 + d1 * s1 + d2 + s0), meshShape = , chipIds = [0]>

#layout = #ttcore.metal_layout<logical_shape = 128x128, dim_alignments = 64x64, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>

// CHECK-LABEL: func.func @grid_selection_preserves_spatial
module attributes {ttcore.device = #any_device} {
  func.func @grid_selection_preserves_spatial()
      -> tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout> {
    %0 = d2m.empty() : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>
    // CHECK: d2m.spatial
    // CHECK-SAME: grid_ranges = #ttcore.core_range_set<[#ttcore.core_range<(0,0), (1,1)>]>
    %1 = d2m.spatial {
      grid_ranges = #ttcore.core_range_set<[#ttcore.core_range<(0, 0), (1, 1)>]>
    } ins() outs(%0 : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>) {
      ^region_0:
        %2 = d2m.generic {
          block_factors = [1, 1],
          grid = #ttcore.grid<2x2>,
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = [#ttcore.iterator_type<parallel>,
                            #ttcore.iterator_type<parallel>],
          threads = [#d2m.thread<unified>]
        } ins() outs(%0 : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>) {
        ^unified0(%cb_out: !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>>>):
          %out = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
          d2m.yield %out : (tensor<2x2x!ttcore.tile<32x32, f32>>)
        } : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>
        d2m.spatial_yield %2 : (tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>)
    } : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>
    return %1 : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>
  }
}
