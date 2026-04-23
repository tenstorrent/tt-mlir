// RUN: ttmlir-opt --split-input-file --ttcore-register-device --d2m-grid-selection %s | FileCheck %s

// d2m-grid-selection: generic inside d2m.spatial uses each region's CoreRange extent for target
// grid sizing (not the full device grid). Square grid_ranges only: pass applies getSquareTargetGrid
// to that extent. Non-origin region output buffers may use d2m.empty VGM so grid CHECKs include maps.
// -----
// Grid 2x2 at origin: generic keeps grid 2x2.
#any_device = #ttcore.device<workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1) -> (0, d0, d1)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1] -> (0, 0, 0, d0 * s1 + d1 * s1 + d2 + s0), meshShape = , chipIds = [0]>
#layout = #ttcore.metal_layout<logical_shape = 128x128, dim_alignments = 64x64, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>
// CHECK-LABEL: func.func @spatial_grid_2x2_origin
module attributes {ttcore.device = #any_device} {
  func.func @spatial_grid_2x2_origin()
      -> tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout> {
    %0 = d2m.empty() : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>
    // CHECK: d2m.spatial
    // CHECK-SAME: grid_ranges = [#ttcore.core_range<(0,0), (1,1)>]
    %1 = d2m.spatial {
      grid_ranges = [#ttcore.core_range<(0, 0), (1, 1)>]
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
          %out = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
          d2m.yield %out : (tensor<2x2x!ttcore.tile<32x32, f32>>)
        } : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>
        d2m.spatial_yield %2 : (tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>)
    } : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>
    // CHECK: d2m.generic
    // CHECK-SAME: grid = #ttcore.grid<2x2>,
    return %1 : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>
  }
}

// -----
// Grid 1x1 at origin: generic keeps grid 1x1.
#any_device = #ttcore.device<workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1) -> (0, d0, d1)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1] -> (0, 0, 0, d0 * s1 + d1 * s1 + d2 + s0), meshShape = , chipIds = [0]>
#layout_1x1 = #ttcore.metal_layout<logical_shape = 128x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>
// CHECK-LABEL: func.func @spatial_grid_1x1_origin
module attributes {ttcore.device = #any_device} {
  func.func @spatial_grid_1x1_origin()
      -> tensor<1x1x4x4x!ttcore.tile<32x32, f32>, #layout_1x1> {
    %0 = d2m.empty() : tensor<1x1x4x4x!ttcore.tile<32x32, f32>, #layout_1x1>
    // CHECK: d2m.spatial
    // CHECK-SAME: grid_ranges = [#ttcore.core_range<(0,0), (0,0)>]
    %1 = d2m.spatial {
      grid_ranges = [#ttcore.core_range<(0, 0), (0, 0)>]
    } ins() outs(%0 : tensor<1x1x4x4x!ttcore.tile<32x32, f32>, #layout_1x1>) {
      ^region_0:
        %2 = d2m.generic {
          block_factors = [1, 1],
          grid = #ttcore.grid<1x1>,
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = [#ttcore.iterator_type<parallel>,
                            #ttcore.iterator_type<parallel>],
          threads = [#d2m.thread<unified>]
        } ins() outs(%0 : tensor<1x1x4x4x!ttcore.tile<32x32, f32>, #layout_1x1>) {
          %out = tensor.empty() : tensor<4x4x!ttcore.tile<32x32, f32>>
          d2m.yield %out : (tensor<4x4x!ttcore.tile<32x32, f32>>)
        } : tensor<1x1x4x4x!ttcore.tile<32x32, f32>, #layout_1x1>
        d2m.spatial_yield %2 : (tensor<1x1x4x4x!ttcore.tile<32x32, f32>, #layout_1x1>)
    } : tensor<1x1x4x4x!ttcore.tile<32x32, f32>, #layout_1x1>
    // CHECK: d2m.generic
    // CHECK-SAME: grid = #ttcore.grid<1x1>,
    return %1 : tensor<1x1x4x4x!ttcore.tile<32x32, f32>, #layout_1x1>
  }
}

// -----
// Two regions: grid_ranges (0,0)-(0,0) and (1,1)-(2,2). Each generic keeps its region grid (1x1, 2x2).
// Non-origin region output buffer uses d2m.empty VGM (physical core (y,x) to virtual shard indices).
#any_device = #ttcore.device<workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1) -> (0, d0, d1)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1] -> (0, 0, 0, d0 * s1 + d1 * s1 + d2 + s0), meshShape = , chipIds = [0]>
#layout_1x1 = #ttcore.metal_layout<logical_shape = 128x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>
#layout = #ttcore.metal_layout<logical_shape = 128x128, dim_alignments = 64x64, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>
#id = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#vgm_11_inv = affine_map<(d0, d1) -> (0, d0 - 1, d1 - 1)>
#vgm_11_fwd = affine_map<(d0, d1, d2, d3) -> (d0 + 1, d1 + 1, d2, d3)>
// CHECK-LABEL: func.func @spatial_multi_region_two_ranges
module attributes {ttcore.device = #any_device} {
  func.func @spatial_multi_region_two_ranges()
      -> (tensor<1x1x4x4x!ttcore.tile<32x32, f32>, #layout_1x1>,
          tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>) {
    %0 = d2m.empty() : tensor<1x1x4x4x!ttcore.tile<32x32, f32>, #layout_1x1>
    %1 = d2m.empty() {virtualGridInverseMapping = #vgm_11_inv, virtualGridForwardMapping = #vgm_11_fwd} : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>
    %view0 = d2m.view_layout %0 remapping = #id : tensor<1x1x4x4x!ttcore.tile<32x32, f32>, #layout_1x1> -> tensor<1x1x4x4x!ttcore.tile<32x32, f32>, #layout_1x1>
    %view1 = d2m.view_layout %1 remapping = #id : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout> -> tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>
    // CHECK: d2m.spatial
    // CHECK-SAME: grid_ranges = [#ttcore.core_range<(0,0), (0,0)>, #ttcore.core_range<(1,1), (2,2)>]
    %2:2 = d2m.spatial {grid_ranges = [#ttcore.core_range<(0, 0), (0, 0)>, #ttcore.core_range<(1, 1), (2, 2)>]}
        ins() outs(%view0, %view1 : tensor<1x1x4x4x!ttcore.tile<32x32, f32>, #layout_1x1>, tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>) {
      ^region_0:
        %3 = d2m.generic {
          block_factors = [1, 1],
          grid = #ttcore.grid<1x1>,
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>],
          threads = [#d2m.thread<unified>]
        } ins() outs(%view0 : tensor<1x1x4x4x!ttcore.tile<32x32, f32>, #layout_1x1>) {
          %out = tensor.empty() : tensor<4x4x!ttcore.tile<32x32, f32>>
          d2m.yield %out : (tensor<4x4x!ttcore.tile<32x32, f32>>)
        } : tensor<1x1x4x4x!ttcore.tile<32x32, f32>, #layout_1x1>
        d2m.spatial_yield %3 : (tensor<1x1x4x4x!ttcore.tile<32x32, f32>, #layout_1x1>)
      }, {
      ^region_1:
        %4 = d2m.generic {
          block_factors = [1, 1],
          grid = #ttcore.grid<2x2>,
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>],
          threads = [#d2m.thread<unified>]
        } ins() outs(%view1 : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>) {
          %out = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
          d2m.yield %out : (tensor<2x2x!ttcore.tile<32x32, f32>>)
        } : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>
        d2m.spatial_yield %4 : (tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>)
    } : tensor<1x1x4x4x!ttcore.tile<32x32, f32>, #layout_1x1>, tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>
    // CHECK: d2m.generic
    // CHECK-SAME: grid = #ttcore.grid<1x1>,
    // CHECK: d2m.generic
    // CHECK-SAME: grid = #ttcore.grid<2x2, virt_to_physical_map = (d0, d1) -> (0, d0 + 1, d1 + 1), physical_to_virt_map = (d0, d1) -> (0, d0 - 1, d1 - 1)>,
    return %2#0, %2#1 : tensor<1x1x4x4x!ttcore.tile<32x32, f32>, #layout_1x1>, tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout>
  }
}

// -----
// Four regions on an 8x8 device: each grid_range is a 4x4 square (quadrant).
// Exercises spatial with more than two regions; each generic keeps grid 4x4.
// Non-origin quadrants use d2m.empty VGM (offset from physical (y,x) to virtual 0..3).
// logical_shape 128x128 with 32x32 tiles: 4x4 grid x 1x1 tiles/core => 4*32 = 128 per axis.
#any_device = #ttcore.device<workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1) -> (0, d0, d1)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1] -> (0, 0, 0, d0 * s1 + d1 * s1 + d2 + s0), meshShape = , chipIds = [0]>
#layout_4x4 = #ttcore.metal_layout<logical_shape = 128x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>
#id_4x4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#vgm_q1_inv = affine_map<(d0, d1) -> (0, d0, d1 - 4)>
#vgm_q1_fwd = affine_map<(d0, d1, d2, d3) -> (d0, d1 + 4, d2, d3)>
#vgm_q2_inv = affine_map<(d0, d1) -> (0, d0 - 4, d1)>
#vgm_q2_fwd = affine_map<(d0, d1, d2, d3) -> (d0 + 4, d1, d2, d3)>
#vgm_q3_inv = affine_map<(d0, d1) -> (0, d0 - 4, d1 - 4)>
#vgm_q3_fwd = affine_map<(d0, d1, d2, d3) -> (d0 + 4, d1 + 4, d2, d3)>
// CHECK-LABEL: func.func @spatial_multi_region_four_quadrants_4x4
module attributes {ttcore.device = #any_device} {
  func.func @spatial_multi_region_four_quadrants_4x4()
      -> (tensor<4x4x1x1x!ttcore.tile<32x32, f32>, #layout_4x4>,
          tensor<4x4x1x1x!ttcore.tile<32x32, f32>, #layout_4x4>,
          tensor<4x4x1x1x!ttcore.tile<32x32, f32>, #layout_4x4>,
          tensor<4x4x1x1x!ttcore.tile<32x32, f32>, #layout_4x4>) {
    %0 = d2m.empty() : tensor<4x4x1x1x!ttcore.tile<32x32, f32>, #layout_4x4>
    %1 = d2m.empty() {virtualGridInverseMapping = #vgm_q1_inv, virtualGridForwardMapping = #vgm_q1_fwd} : tensor<4x4x1x1x!ttcore.tile<32x32, f32>, #layout_4x4>
    %2 = d2m.empty() {virtualGridInverseMapping = #vgm_q2_inv, virtualGridForwardMapping = #vgm_q2_fwd} : tensor<4x4x1x1x!ttcore.tile<32x32, f32>, #layout_4x4>
    %3 = d2m.empty() {virtualGridInverseMapping = #vgm_q3_inv, virtualGridForwardMapping = #vgm_q3_fwd} : tensor<4x4x1x1x!ttcore.tile<32x32, f32>, #layout_4x4>
    %view0 = d2m.view_layout %0 remapping = #id_4x4 : tensor<4x4x1x1x!ttcore.tile<32x32, f32>, #layout_4x4> -> tensor<4x4x1x1x!ttcore.tile<32x32, f32>, #layout_4x4>
    %view1 = d2m.view_layout %1 remapping = #id_4x4 : tensor<4x4x1x1x!ttcore.tile<32x32, f32>, #layout_4x4> -> tensor<4x4x1x1x!ttcore.tile<32x32, f32>, #layout_4x4>
    %view2 = d2m.view_layout %2 remapping = #id_4x4 : tensor<4x4x1x1x!ttcore.tile<32x32, f32>, #layout_4x4> -> tensor<4x4x1x1x!ttcore.tile<32x32, f32>, #layout_4x4>
    %view3 = d2m.view_layout %3 remapping = #id_4x4 : tensor<4x4x1x1x!ttcore.tile<32x32, f32>, #layout_4x4> -> tensor<4x4x1x1x!ttcore.tile<32x32, f32>, #layout_4x4>
    // CHECK: d2m.spatial
    // CHECK-SAME: #ttcore.core_range<(0,0), (3,3)>
    // CHECK-SAME: #ttcore.core_range<(0,4), (3,7)>
    // CHECK-SAME: #ttcore.core_range<(4,0), (7,3)>
    // CHECK-SAME: #ttcore.core_range<(4,4), (7,7)>
    %4:4 = d2m.spatial {grid_ranges = [#ttcore.core_range<(0, 0), (3, 3)>, #ttcore.core_range<(0, 4), (3, 7)>, #ttcore.core_range<(4, 0), (7, 3)>, #ttcore.core_range<(4, 4), (7, 7)>]}
        ins() outs(%view0, %view1, %view2, %view3 : tensor<4x4x1x1x!ttcore.tile<32x32, f32>, #layout_4x4>, tensor<4x4x1x1x!ttcore.tile<32x32, f32>, #layout_4x4>, tensor<4x4x1x1x!ttcore.tile<32x32, f32>, #layout_4x4>, tensor<4x4x1x1x!ttcore.tile<32x32, f32>, #layout_4x4>) {
      ^region_0:
        %g0 = d2m.generic {
          block_factors = [1, 1],
          grid = #ttcore.grid<4x4>,
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>],
          threads = [#d2m.thread<unified>]
        } ins() outs(%view0 : tensor<4x4x1x1x!ttcore.tile<32x32, f32>, #layout_4x4>) {
          %out = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, f32>>
          d2m.yield %out : (tensor<1x1x!ttcore.tile<32x32, f32>>)
        } : tensor<4x4x1x1x!ttcore.tile<32x32, f32>, #layout_4x4>
        d2m.spatial_yield %g0 : (tensor<4x4x1x1x!ttcore.tile<32x32, f32>, #layout_4x4>)
      }, {
      ^region_1:
        %g1 = d2m.generic {
          block_factors = [1, 1],
          grid = #ttcore.grid<4x4>,
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>],
          threads = [#d2m.thread<unified>]
        } ins() outs(%view1 : tensor<4x4x1x1x!ttcore.tile<32x32, f32>, #layout_4x4>) {
          %out = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, f32>>
          d2m.yield %out : (tensor<1x1x!ttcore.tile<32x32, f32>>)
        } : tensor<4x4x1x1x!ttcore.tile<32x32, f32>, #layout_4x4>
        d2m.spatial_yield %g1 : (tensor<4x4x1x1x!ttcore.tile<32x32, f32>, #layout_4x4>)
      }, {
      ^region_2:
        %g2 = d2m.generic {
          block_factors = [1, 1],
          grid = #ttcore.grid<4x4>,
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>],
          threads = [#d2m.thread<unified>]
        } ins() outs(%view2 : tensor<4x4x1x1x!ttcore.tile<32x32, f32>, #layout_4x4>) {
          %out = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, f32>>
          d2m.yield %out : (tensor<1x1x!ttcore.tile<32x32, f32>>)
        } : tensor<4x4x1x1x!ttcore.tile<32x32, f32>, #layout_4x4>
        d2m.spatial_yield %g2 : (tensor<4x4x1x1x!ttcore.tile<32x32, f32>, #layout_4x4>)
      }, {
      ^region_3:
        %g3 = d2m.generic {
          block_factors = [1, 1],
          grid = #ttcore.grid<4x4>,
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
          iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>],
          threads = [#d2m.thread<unified>]
        } ins() outs(%view3 : tensor<4x4x1x1x!ttcore.tile<32x32, f32>, #layout_4x4>) {
          %out = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, f32>>
          d2m.yield %out : (tensor<1x1x!ttcore.tile<32x32, f32>>)
        } : tensor<4x4x1x1x!ttcore.tile<32x32, f32>, #layout_4x4>
        d2m.spatial_yield %g3 : (tensor<4x4x1x1x!ttcore.tile<32x32, f32>, #layout_4x4>)
    } : tensor<4x4x1x1x!ttcore.tile<32x32, f32>, #layout_4x4>, tensor<4x4x1x1x!ttcore.tile<32x32, f32>, #layout_4x4>, tensor<4x4x1x1x!ttcore.tile<32x32, f32>, #layout_4x4>, tensor<4x4x1x1x!ttcore.tile<32x32, f32>, #layout_4x4>
    // CHECK: d2m.generic
    // CHECK-SAME: grid = #ttcore.grid<4x4>,
    // CHECK: d2m.generic
    // CHECK-SAME: grid = #ttcore.grid<4x4, virt_to_physical_map = (d0, d1) -> (0, d0, d1 + 4), physical_to_virt_map = (d0, d1) -> (0, d0, d1 - 4)>,
    // CHECK: d2m.generic
    // CHECK-SAME: grid = #ttcore.grid<4x4, virt_to_physical_map = (d0, d1) -> (0, d0 + 4, d1), physical_to_virt_map = (d0, d1) -> (0, d0 - 4, d1)>,
    // CHECK: d2m.generic
    // CHECK-SAME: grid = #ttcore.grid<4x4, virt_to_physical_map = (d0, d1) -> (0, d0 + 4, d1 + 4), physical_to_virt_map = (d0, d1) -> (0, d0 - 4, d1 - 4)>,
    return %4#0, %4#1, %4#2, %4#3 : tensor<4x4x1x1x!ttcore.tile<32x32, f32>, #layout_4x4>, tensor<4x4x1x1x!ttcore.tile<32x32, f32>, #layout_4x4>, tensor<4x4x1x1x!ttcore.tile<32x32, f32>, #layout_4x4>, tensor<4x4x1x1x!ttcore.tile<32x32, f32>, #layout_4x4>
  }
}

// -----
// Generic grid mapping from cast VGM: when spatial has non-origin grid_ranges
// and the generic output is ttir.ttnn_metal_layout_cast with
// virtual_grid_inverse_mapping (and virtual_grid_forward_mapping), the generic
// gets grid with that mapping after GridSelection.
#any_device = #ttcore.device<workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1) -> (0, d0, d1)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1] -> (0, 0, 0, d0 * s1 + d1 * s1 + d2 + s0), meshShape = , chipIds = [0]>
#l1 = #ttnn.buffer_type<l1>
#ttnn_layout_vgm = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1, virt_to_physical_map = (d0, d1) -> (0, d0 + 2, d1 + 2), physical_to_virt_map = (d0, d1, d2) -> (d1 - 2, d2 - 2)>, memref<2x2x!ttcore.tile<32x32, f32>, #l1>, <block_sharded>>
#layout_cast = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>
#map_vgm_inv = affine_map<(d0, d1) -> (0, d0 - 2, d1 - 2)>
#map_vgm_fwd = affine_map<(d0, d1, d2, d3) -> (d0 + 2, d1 + 2, d2, d3)>
// CHECK-LABEL: func.func @spatial_non_origin_generic_grid_from_cast_vgm
module attributes {ttcore.device = #any_device} {
  func.func @spatial_non_origin_generic_grid_from_cast_vgm()
      -> tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_cast> {
    %0 = d2m.empty() : tensor<64x64xf32, #ttnn_layout_vgm>
    %1 = ttir.ttnn_metal_layout_cast %0 {virtual_grid_forward_mapping = #map_vgm_fwd, virtual_grid_inverse_mapping = #map_vgm_inv} : tensor<64x64xf32, #ttnn_layout_vgm> -> tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_cast>
    %2 = d2m.spatial {
      grid_ranges = [#ttcore.core_range<(2, 2), (2, 2)>]
    } ins() outs(%1 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_cast>) {
    ^region_0:
      %3 = d2m.generic {
        block_factors = [1, 1],
        grid = #ttcore.grid<1x1>,
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = [#ttcore.iterator_type<parallel>,
                          #ttcore.iterator_type<parallel>],
        threads = [#d2m.thread<unified>]
      } ins() outs(%1 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_cast>) {
        %out = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
        d2m.yield %out : (tensor<2x2x!ttcore.tile<32x32, f32>>)
      } : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_cast>
      d2m.spatial_yield %3 : (tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_cast>)
    } : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_cast>
    // CHECK: d2m.generic
    // CHECK-SAME: grid = #ttcore.grid<1x1, virt_to_physical_map = (d0, d1) -> (0, d0 + 2, d1 + 2), physical_to_virt_map = (d0, d1) -> (0, d0 - 2, d1 - 2)>,
    return %2 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_cast>
  }
}
