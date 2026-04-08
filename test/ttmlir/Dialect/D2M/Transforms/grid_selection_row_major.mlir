// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m --d2m-materialize-view-returns --d2m-grid-selection --canonicalize --split-input-file %s | FileCheck %s

#any_device = #ttcore.device<workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1] -> (0, 0, 0, d0 * s1 + d1 * s1 + d2 + s0), meshShape = , chipIds = [0]>
module attributes {ttcore.device = #any_device} {
  func.func @test_grid_selection_row_major(%arg0: tensor<1x64x64x32xf32>) -> tensor<1x64x64x32xf32> {
    %0 = d2m.empty() : tensor<1x64x64x32xf32>
    // Verify D2MGridSelection optimizes to 1x32x2x1 grids
    // CHECK: d2m.generic {{.*}}grid = #ttcore.grid<1x32x2x1,
    %1 = d2m.empty() : tensor<1x1x1x1x1x64x64x32xf32, #ttcore.metal_layout<logical_shape = 1x64x64x32, dim_alignments = 1x1x32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, undef, l1, sharded>>
    %2 = d2m.to_layout %arg0, %1 : tensor<1x64x64x32xf32> into tensor<1x1x1x1x1x64x64x32xf32, #ttcore.metal_layout<logical_shape = 1x64x64x32, dim_alignments = 1x1x32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, undef, l1, sharded>> -> tensor<1x1x1x1x1x64x64x32xf32, #ttcore.metal_layout<logical_shape = 1x64x64x32, dim_alignments = 1x1x32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, undef, l1, sharded>>
    %view = d2m.view_layout %2 remapping = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d2, d1, d3, d4, d6, d5, d7)> : tensor<1x1x1x1x1x64x64x32xf32, #ttcore.metal_layout<logical_shape = 1x64x64x32, dim_alignments = 1x1x32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, undef, l1, sharded>> -> tensor<1x1x1x1x1x64x64x32xf32, #ttcore.metal_layout<logical_shape = 1x64x64x32, dim_alignments = 1x1x32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, undef, l1, sharded>>
    %3 = d2m.empty() : tensor<1x64x64x32xf32>
    %4 = d2m.to_layout %view, %3 : tensor<1x1x1x1x1x64x64x32xf32, #ttcore.metal_layout<logical_shape = 1x64x64x32, dim_alignments = 1x1x32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, undef, l1, sharded>> into tensor<1x64x64x32xf32> -> tensor<1x64x64x32xf32>
    return %4 : tensor<1x64x64x32xf32>
  }
}
