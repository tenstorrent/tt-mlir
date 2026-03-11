// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for d2m.spatial operation verification.
//
// SpatialOp::verify() checks (in order):
// 1. grid_ranges has at least one CoreRange
// 2. number of CoreRanges matches number of Regions
// 3. each region contains exactly one d2m.generic
// 4. generic grid is 2D (grid shape size == 2)
// 5. when grid mapping is empty: grid shape <= region shape
// 6. when grid mapping is present: virtual grid [0,0]-[gridY-1,gridX-1] contained
//    in region virtual bbox (inverse map applied to CoreRange)

#layout = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>

// 1. grid_ranges must contain at least one CoreRange
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

// 2. number of CoreRanges must match number of Regions (two ranges, one region)
// CHECK: error: 'd2m.spatial' op number of CoreRanges (2) must match the number of Regions (1)

module {
  func.func @spatial_core_ranges_mismatch_regions(
      %arg0: tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_2>)
      -> tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_2> {
    %0 = d2m.spatial {
      grid_ranges = #ttcore.core_range_set<[
        #ttcore.core_range<(1, 1), (1, 1)>,
        #ttcore.core_range<(2, 2), (3, 4)>
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

// -----

#layout_3 = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>

// 3a. each region must contain exactly one d2m.generic (got 0)
// CHECK: error: 'd2m.spatial' op each region must contain exactly one d2m.generic op, got 0

module {
  func.func @spatial_region_no_generic(
      %arg0: tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_3>)
      -> tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_3> {
    %0 = d2m.spatial {
      grid_ranges = #ttcore.core_range_set<[#ttcore.core_range<(1, 1), (1, 1)>]>
    } ins() outs(%arg0 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_3>) {
      ^region_0:
        d2m.spatial_yield %arg0 : (tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_3>)
    } : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_3>
    return %0 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_3>
  }
}

// -----

#layout_3b = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>

// 3b. each region must contain exactly one d2m.generic (got 2)
// CHECK: error: 'd2m.spatial' op each region must contain exactly one d2m.generic op, got 2

module {
  func.func @spatial_region_two_generics(
      %arg0: tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_3b>)
      -> tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_3b> {
    %0 = d2m.spatial {
      grid_ranges = #ttcore.core_range_set<[#ttcore.core_range<(1, 1), (1, 1)>]>
    } ins() outs(%arg0 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_3b>) {
      ^region_0:
        %1 = d2m.generic {
          block_factors = [1, 1],
          grid = #ttcore.grid<1x1>,
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = [#ttcore.iterator_type<parallel>,
                            #ttcore.iterator_type<parallel>],
          threads = [#d2m.thread<unified>]
        } ins() outs(%arg0 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_3b>) {
        ^unified0(%cb_out: !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>>>):
          %out = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
          d2m.yield %out : (tensor<2x2x!ttcore.tile<32x32, f32>>)
        } : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_3b>
        %2 = d2m.generic {
          block_factors = [1, 1],
          grid = #ttcore.grid<1x1>,
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = [#ttcore.iterator_type<parallel>,
                            #ttcore.iterator_type<parallel>],
          threads = [#d2m.thread<unified>]
        } ins() outs(%arg0 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_3b>) {
        ^unified0(%cb_out: !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>>>):
          %out = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
          d2m.yield %out : (tensor<2x2x!ttcore.tile<32x32, f32>>)
        } : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_3b>
        d2m.spatial_yield %2 : (tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_3b>)
    } : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_3b>
    return %0 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_3b>
  }
}

// -----

#layout_4 = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>

// 4. generic grid must be 2D
// CHECK: error: 'd2m.spatial' op d2m.generic inside d2m.spatial: only 2D grid is considered for now, got 3D

module {
  func.func @spatial_generic_non_2d_grid(
      %arg0: tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_4>)
      -> tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_4> {
    %0 = d2m.spatial {
      grid_ranges = #ttcore.core_range_set<[#ttcore.core_range<(1, 1), (1, 1)>]>
    } ins() outs(%arg0 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_4>) {
      ^region_0:
        %1 = d2m.generic {
          block_factors = [1, 1, 1],
          grid = #ttcore.grid<1x1x1>,
          indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
          iterator_types = [#ttcore.iterator_type<parallel>,
                            #ttcore.iterator_type<parallel>,
                            #ttcore.iterator_type<parallel>],
          threads = [#d2m.thread<unified>]
        } ins() outs(%arg0 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_4>) {
        ^unified0(%cb_out: !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>>>):
          %out = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
          d2m.yield %out : (tensor<2x2x!ttcore.tile<32x32, f32>>)
        } : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_4>
        d2m.spatial_yield %1 : (tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_4>)
    } : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_4>
    return %0 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_4>
  }
}

// -----

#layout_5 = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>

// 5. mapping empty: grid shape must not exceed region shape (CoreRange (1,1)-(2,4), shape 2x4)
// CHECK: error: 'd2m.spatial' op generic op grid shape [3, 5] exceeds region CoreRange shape [2, 4]

module {
  func.func @spatial_grid_exceeds_core_range(
      %arg0: tensor<3x5x2x2x!ttcore.tile<32x32, f32>, #layout_5>)
      -> tensor<3x5x2x2x!ttcore.tile<32x32, f32>, #layout_5> {
    %0 = d2m.spatial {
      grid_ranges = #ttcore.core_range_set<[#ttcore.core_range<(1, 1), (2, 4)>]>
    } ins() outs(%arg0 : tensor<3x5x2x2x!ttcore.tile<32x32, f32>, #layout_5>) {
      ^region_0:
        %1 = d2m.generic {
          block_factors = [1, 1],
          grid = #ttcore.grid<3x5>,
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = [#ttcore.iterator_type<parallel>,
                            #ttcore.iterator_type<parallel>],
          threads = [#d2m.thread<unified>]
        } ins() outs(%arg0 : tensor<3x5x2x2x!ttcore.tile<32x32, f32>, #layout_5>) {
        ^unified0(%cb_out: !d2m.cb<tensor<3x5x!ttcore.tile<32x32, f32>>>):
          %out = tensor.empty() : tensor<3x5x!ttcore.tile<32x32, f32>>
          d2m.yield %out : (tensor<3x5x!ttcore.tile<32x32, f32>>)
        } : tensor<3x5x2x2x!ttcore.tile<32x32, f32>, #layout_5>
        d2m.spatial_yield %1 : (tensor<3x5x2x2x!ttcore.tile<32x32, f32>, #layout_5>)
    } : tensor<3x5x2x2x!ttcore.tile<32x32, f32>, #layout_5>
    return %0 : tensor<3x5x2x2x!ttcore.tile<32x32, f32>, #layout_5>
  }
}

// -----

#layout_6b = #ttcore.metal_layout<logical_shape = 128x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>

// 6. mapping present: virtual grid contained in region virtual range (identity map)
// Region (2,2)-(2,2) -> virtual bbox (2,2)-(2,2).
// CHECK: error: 'd2m.spatial' op generic op grid not contained in region grid_ranges [2, 2] to [2, 2]

module {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1] -> (0, 0, 0, d0 * s1 + d1 * s1 + d2 + s0), meshShape = , chipIds = [0]>
  func.func @spatial_physical_grid_not_contained()
      -> tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_6b> {
    %0 = d2m.empty() {virtualGridInverseMapping = affine_map<(d0, d1) -> (0, d0, d1)>}
        : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_6b>
    %1 = d2m.spatial {
      grid_ranges = #ttcore.core_range_set<[#ttcore.core_range<(2, 2), (2, 2)>]>
    } ins() outs(%0 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_6b>) {
      ^region_0:
        %2 = d2m.generic {
          block_factors = [1, 1],
          grid = #ttcore.grid<1x1, (d0, d1) -> (0, d0, d1)>,
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = [#ttcore.iterator_type<parallel>,
                            #ttcore.iterator_type<parallel>],
          threads = [#d2m.thread<unified>]
        } ins() outs(%0 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_6b>) {
        ^unified0(%cb_out: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>):
          %out = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, f32>>
          d2m.yield %out : (tensor<1x1x!ttcore.tile<32x32, f32>>)
        } : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_6b>
        d2m.spatial_yield %2 : (tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_6b>)
    } : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_6b>
    return %1 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_6b>
  }
}

// -----

#layout_6b_2 = #ttcore.metal_layout<logical_shape = 128x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>

// GenericOp verify (not SpatialOp): same pipeline; generic with grid mapping requires output with VGM.
// CHECK: error: 'd2m.generic' op grid has an inverse map but output operand does not have a VGM

module {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1] -> (0, 0, 0, d0 * s1 + d1 * s1 + d2 + s0), meshShape = , chipIds = [0]>
  func.func @spatial_grid_mapping_output_no_vgm(
      %arg0: tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout_6b_2>)
      -> tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout_6b_2> {
    %0 = d2m.spatial {
      grid_ranges = #ttcore.core_range_set<[#ttcore.core_range<(0, 0), (2, 4)>]>
    } ins() outs(%arg0 : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout_6b_2>) {
      ^region_0:
        %1 = d2m.generic {
          block_factors = [1, 1],
          grid = #ttcore.grid<2x2, (d0, d1) -> (0, d0, d1)>,
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = [#ttcore.iterator_type<parallel>,
                            #ttcore.iterator_type<parallel>],
          threads = [#d2m.thread<unified>]
        } ins() outs(%arg0 : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout_6b_2>) {
        ^unified0(%cb_out: !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>>>):
          %out = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
          d2m.yield %out : (tensor<2x2x!ttcore.tile<32x32, f32>>)
        } : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout_6b_2>
        d2m.spatial_yield %1 : (tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout_6b_2>)
    } : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout_6b_2>
    return %0 : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout_6b_2>
  }
}

// -----

#layout_6b_3 = #ttcore.metal_layout<logical_shape = 128x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>

// 6. mapping present: virtual range containment with offset map (d0-1,d1-1)
// Region (2,2)-(2,2) -> virtual bbox (1,1)-(1,1).
// CHECK: error: 'd2m.spatial' op generic op grid not contained in region grid_ranges [2, 2] to [2, 2]

module {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1] -> (0, 0, 0, d0 * s1 + d1 * s1 + d2 + s0), meshShape = , chipIds = [0]>
  func.func @spatial_physical_grid_offset_not_contained()
      -> tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_6b_3> {
    %0 = d2m.empty() {virtualGridInverseMapping = affine_map<(d0, d1) -> (0, d0 - 1, d1 - 1)>}
        : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_6b_3>
    %1 = d2m.spatial {
      grid_ranges = #ttcore.core_range_set<[#ttcore.core_range<(2, 2), (2, 2)>]>
    } ins() outs(%0 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_6b_3>) {
      ^region_0:
        %2 = d2m.generic {
          block_factors = [1, 1],
          grid = #ttcore.grid<1x1, (d0, d1) -> (0, d0 - 1, d1 - 1)>,
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = [#ttcore.iterator_type<parallel>,
                            #ttcore.iterator_type<parallel>],
          threads = [#d2m.thread<unified>]
        } ins() outs(%0 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_6b_3>) {
        ^unified0(%cb_out: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>):
          %out = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, f32>>
          d2m.yield %out : (tensor<1x1x!ttcore.tile<32x32, f32>>)
        } : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_6b_3>
        d2m.spatial_yield %2 : (tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_6b_3>)
    } : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_6b_3>
    return %1 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_6b_3>
  }
}

// -----

#layout_6b_4 = #ttcore.metal_layout<logical_shape = 128x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>

// 6. mapping present: only Y axis out. Region (2,0)-(2,3) -> virtual (2,0)-(2,3); grid [0,0]-[1,1] has Y 0,1 not in [2,2].
// CHECK: error: 'd2m.spatial' op generic op grid not contained in region grid_ranges [2, 0] to [2, 3]

module {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1] -> (0, 0, 0, d0 * s1 + d1 * s1 + d2 + s0), meshShape = , chipIds = [0]>
  func.func @spatial_grid_not_contained_y_only()
      -> tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout_6b_4> {
    %0 = d2m.empty() {virtualGridInverseMapping = affine_map<(d0, d1) -> (0, d0, d1)>}
        : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout_6b_4>
    %1 = d2m.spatial {
      grid_ranges = #ttcore.core_range_set<[#ttcore.core_range<(2, 0), (2, 3)>]>
    } ins() outs(%0 : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout_6b_4>) {
      ^region_0:
        %2 = d2m.generic {
          block_factors = [1, 1],
          grid = #ttcore.grid<2x2, (d0, d1) -> (0, d0, d1)>,
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = [#ttcore.iterator_type<parallel>,
                            #ttcore.iterator_type<parallel>],
          threads = [#d2m.thread<unified>]
        } ins() outs(%0 : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout_6b_4>) {
        ^unified0(%cb_out: !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>>>):
          %out = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
          d2m.yield %out : (tensor<2x2x!ttcore.tile<32x32, f32>>)
        } : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout_6b_4>
        d2m.spatial_yield %2 : (tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout_6b_4>)
    } : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout_6b_4>
    return %1 : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout_6b_4>
  }
}

// -----

#layout_6b_5 = #ttcore.metal_layout<logical_shape = 128x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>

// 6. mapping present: only X axis out. Region (0,2)-(3,2) -> virtual (0,2)-(3,2); grid [0,0]-[1,1] has X 0,1 not in [2,2].
// CHECK: error: 'd2m.spatial' op generic op grid not contained in region grid_ranges [0, 2] to [3, 2]

module {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1] -> (0, 0, 0, d0 * s1 + d1 * s1 + d2 + s0), meshShape = , chipIds = [0]>
  func.func @spatial_grid_not_contained_x_only()
      -> tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout_6b_5> {
    %0 = d2m.empty() {virtualGridInverseMapping = affine_map<(d0, d1) -> (0, d0, d1)>}
        : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout_6b_5>
    %1 = d2m.spatial {
      grid_ranges = #ttcore.core_range_set<[#ttcore.core_range<(0, 2), (3, 2)>]>
    } ins() outs(%0 : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout_6b_5>) {
      ^region_0:
        %2 = d2m.generic {
          block_factors = [1, 1],
          grid = #ttcore.grid<2x2, (d0, d1) -> (0, d0, d1)>,
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = [#ttcore.iterator_type<parallel>,
                            #ttcore.iterator_type<parallel>],
          threads = [#d2m.thread<unified>]
        } ins() outs(%0 : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout_6b_5>) {
        ^unified0(%cb_out: !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>>>):
          %out = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
          d2m.yield %out : (tensor<2x2x!ttcore.tile<32x32, f32>>)
        } : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout_6b_5>
        d2m.spatial_yield %2 : (tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout_6b_5>)
    } : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout_6b_5>
    return %1 : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout_6b_5>
  }
}

// -----

#layout_6b_6 = #ttcore.metal_layout<logical_shape = 128x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>

// 6. mapping present: map (d0,d1)->(2-d0,d1) reverses Y; Region (0,0)-(1,0) -> virtual (1,0)-(2,0); grid [0,0] not in that range.
// CHECK: error: 'd2m.spatial' op generic op grid not contained in region grid_ranges [0, 0] to [1, 0]

module {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1] -> (0, 0, 0, d0 * s1 + d1 * s1 + d2 + s0), meshShape = , chipIds = [0]>
  func.func @spatial_grid_not_contained_reversed_y()
      -> tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_6b_6> {
    %0 = d2m.empty() {virtualGridInverseMapping = affine_map<(d0, d1) -> (0, 2 - d0, d1)>}
        : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_6b_6>
    %1 = d2m.spatial {
      grid_ranges = #ttcore.core_range_set<[#ttcore.core_range<(0, 0), (1, 0)>]>
    } ins() outs(%0 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_6b_6>) {
      ^region_0:
        %2 = d2m.generic {
          block_factors = [1, 1],
          grid = #ttcore.grid<1x1, (d0, d1) -> (0, 2 - d0, d1)>,
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = [#ttcore.iterator_type<parallel>,
                            #ttcore.iterator_type<parallel>],
          threads = [#d2m.thread<unified>]
        } ins() outs(%0 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_6b_6>) {
        ^unified0(%cb_out: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>):
          %out = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, f32>>
          d2m.yield %out : (tensor<1x1x!ttcore.tile<32x32, f32>>)
        } : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_6b_6>
        d2m.spatial_yield %2 : (tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_6b_6>)
    } : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_6b_6>
    return %1 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_6b_6>
  }
}

// -----

#layout_6b_7 = #ttcore.metal_layout<logical_shape = 128x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>

// 6. mapping (d0,d1)->(d1,d0) swap: Region (0,2)-(0,2) -> virtual (2,0)-(2,0).
// CHECK: error: 'd2m.spatial' op generic op grid not contained in region grid_ranges [0, 2] to [0, 2]

module {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1] -> (0, 0, 0, d0 * s1 + d1 * s1 + d2 + s0), meshShape = , chipIds = [0]>
  func.func @spatial_grid_not_contained_swap_map()
      -> tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_6b_7> {
    %0 = d2m.empty() {virtualGridInverseMapping = affine_map<(d0, d1) -> (0, d1, d0)>}
        : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_6b_7>
    %1 = d2m.spatial {
      grid_ranges = #ttcore.core_range_set<[#ttcore.core_range<(0, 2), (0, 2)>]>
    } ins() outs(%0 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_6b_7>) {
      ^region_0:
        %2 = d2m.generic {
          block_factors = [1, 1],
          grid = #ttcore.grid<1x1, (d0, d1) -> (0, d1, d0)>,
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = [#ttcore.iterator_type<parallel>,
                            #ttcore.iterator_type<parallel>],
          threads = [#d2m.thread<unified>]
        } ins() outs(%0 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_6b_7>) {
        ^unified0(%cb_out: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>):
          %out = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, f32>>
          d2m.yield %out : (tensor<1x1x!ttcore.tile<32x32, f32>>)
        } : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_6b_7>
        d2m.spatial_yield %2 : (tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_6b_7>)
    } : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_6b_7>
    return %1 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_6b_7>
  }
}

// -----

#layout_6b_8 = #ttcore.metal_layout<logical_shape = 128x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>

// 6. mapping (d0,d1)->(d0,d1-1) offset on X: Region (1,1)-(2,1) -> virtual (1,0)-(2,0).
// CHECK: error: 'd2m.spatial' op generic op grid not contained in region grid_ranges [1, 1] to [2, 1]

module {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1] -> (0, 0, 0, d0 * s1 + d1 * s1 + d2 + s0), meshShape = , chipIds = [0]>
  func.func @spatial_grid_not_contained_offset_x()
      -> tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_6b_8> {
    %0 = d2m.empty() {virtualGridInverseMapping = affine_map<(d0, d1) -> (0, d0, d1 - 1)>}
        : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_6b_8>
    %1 = d2m.spatial {
      grid_ranges = #ttcore.core_range_set<[#ttcore.core_range<(1, 1), (2, 1)>]>
    } ins() outs(%0 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_6b_8>) {
      ^region_0:
        %2 = d2m.generic {
          block_factors = [1, 1],
          grid = #ttcore.grid<1x1, (d0, d1) -> (0, d0, d1 - 1)>,
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = [#ttcore.iterator_type<parallel>,
                            #ttcore.iterator_type<parallel>],
          threads = [#d2m.thread<unified>]
        } ins() outs(%0 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_6b_8>) {
        ^unified0(%cb_out: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>):
          %out = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, f32>>
          d2m.yield %out : (tensor<1x1x!ttcore.tile<32x32, f32>>)
        } : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_6b_8>
        d2m.spatial_yield %2 : (tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_6b_8>)
    } : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_6b_8>
    return %1 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_6b_8>
  }
}

// -----

#layout_6b_9 = #ttcore.metal_layout<logical_shape = 128x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>

// 6. mapping (d0,d1)->(2*d0,d1) scale Y: Region (0,0)-(1,0) -> virtual (0,0)-(2,0); grid 2x2 needs X in [0,1] but virtual X only [0,0].
// CHECK: error: 'd2m.spatial' op generic op grid not contained in region grid_ranges [0, 0] to [1, 0]

module {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1] -> (0, 0, 0, d0 * s1 + d1 * s1 + d2 + s0), meshShape = , chipIds = [0]>
  func.func @spatial_grid_not_contained_scale_y()
      -> tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout_6b_9> {
    %0 = d2m.empty() {virtualGridInverseMapping = affine_map<(d0, d1) -> (0, 2 * d0, d1)>}
        : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout_6b_9>
    %1 = d2m.spatial {
      grid_ranges = #ttcore.core_range_set<[#ttcore.core_range<(0, 0), (1, 0)>]>
    } ins() outs(%0 : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout_6b_9>) {
      ^region_0:
        %2 = d2m.generic {
          block_factors = [1, 1],
          grid = #ttcore.grid<2x2, (d0, d1) -> (0, 2 * d0, d1)>,
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = [#ttcore.iterator_type<parallel>,
                            #ttcore.iterator_type<parallel>],
          threads = [#d2m.thread<unified>]
        } ins() outs(%0 : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout_6b_9>) {
        ^unified0(%cb_out: !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>>>):
          %out = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
          d2m.yield %out : (tensor<2x2x!ttcore.tile<32x32, f32>>)
        } : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout_6b_9>
        d2m.spatial_yield %2 : (tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout_6b_9>)
    } : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout_6b_9>
    return %1 : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout_6b_9>
  }
}
