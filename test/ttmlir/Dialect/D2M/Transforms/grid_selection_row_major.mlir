// RUN: ttmlir-opt --ttcore-register-device --ttir-to-d2m --d2m-grid-selection --canonicalize --split-input-file %s | FileCheck %s

#any_device = #ttcore.device<workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1] -> (0, 0, 0, d0 * s1 + d1 * s1 + d2 + s0), meshShape = , chipIds = [0]>
module attributes {ttcore.device = #any_device} {
  func.func @test_grid_selection_row_major(%arg0: tensor<1x64x64x32xf32>) -> tensor<1x64x64x32xf32> {
    %0 = d2m.empty() : tensor<1x64x64x32xf32>
    // Verify D2MGridSelection optimizes to 1x1x2x1 grids (not 1x1x8x8)
    // CHECK: d2m.generic {{.*}} grid = #ttcore.grid<1x32x2x1>
    %1 = d2m.empty() : tensor<1x1x1x1x1x64x64x32xf32, #ttcore.metal_layout<logical_shape = 1x64x64x32, dim_alignments = 1x1x32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, undef, l1, sharded>>
    %2 = d2m.to_layout %arg0, %1 : tensor<1x64x64x32xf32> into tensor<1x1x1x1x1x64x64x32xf32, #ttcore.metal_layout<logical_shape = 1x64x64x32, dim_alignments = 1x1x32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, undef, l1, sharded>> -> tensor<1x1x1x1x1x64x64x32xf32, #ttcore.metal_layout<logical_shape = 1x64x64x32, dim_alignments = 1x1x32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, undef, l1, sharded>>
    %3 = d2m.empty() : tensor<1x1x1x1x1x64x64x32xf32, #ttcore.metal_layout<logical_shape = 1x64x64x32, dim_alignments = 1x1x32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, undef, l1, sharded>>
    %4 = d2m.to_layout %0, %3 : tensor<1x64x64x32xf32> into tensor<1x1x1x1x1x64x64x32xf32, #ttcore.metal_layout<logical_shape = 1x64x64x32, dim_alignments = 1x1x32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, undef, l1, sharded>> -> tensor<1x1x1x1x1x64x64x32xf32, #ttcore.metal_layout<logical_shape = 1x64x64x32, dim_alignments = 1x1x32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, undef, l1, sharded>>
    %5 = d2m.empty() : tensor<1x1x1x1x1x64x64x32xf32, #ttcore.metal_layout<logical_shape = 1x64x64x32, dim_alignments = 1x1x32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, undef, l1, sharded>>
    %stream = "d2m.stream_layout"(%2, %5) <{remapping = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d2, d1, d3, d4, d6, d5, d7)>}> : (tensor<1x1x1x1x1x64x64x32xf32, #ttcore.metal_layout<logical_shape = 1x64x64x32, dim_alignments = 1x1x32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, undef, l1, sharded>>, tensor<1x1x1x1x1x64x64x32xf32, #ttcore.metal_layout<logical_shape = 1x64x64x32, dim_alignments = 1x1x32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, undef, l1, sharded>>) -> tensor<1x1x1x1x1x64x64x32xf32, #ttcore.metal_layout<logical_shape = 1x64x64x32, dim_alignments = 1x1x32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, undef, l1, sharded>>
    %6 = d2m.generic {block_factors = [1, 1, 1, 1], grid = #ttcore.grid<1x1x1x1>, indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>], threads = [#d2m.thread<datamovement>]}
        ins(%stream : tensor<1x1x1x1x1x64x64x32xf32, #ttcore.metal_layout<logical_shape = 1x64x64x32, dim_alignments = 1x1x32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, undef, l1, sharded>>)
        outs(%4 : tensor<1x1x1x1x1x64x64x32xf32, #ttcore.metal_layout<logical_shape = 1x64x64x32, dim_alignments = 1x1x32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, undef, l1, sharded>>)  {
    ^unified0(%cb0: !d2m.cb<tensor<1x64x64x32xf32>>, %cb1: !d2m.cb<tensor<1x64x64x32xf32>>):
      %i = d2m.block_index(0) : index
      %j = d2m.block_index(1) : index
      %k = d2m.block_index(2) : index
      %l = d2m.block_index(3) : index
      %buffer = tensor.empty() : tensor<1x64x64x32xf32>
      %9 = d2m.remote_load %buffer %stream[%i, %j, %k, %l]
        : tensor<1x64x64x32xf32>, tensor<1x1x1x1x1x64x64x32xf32, #ttcore.metal_layout<logical_shape = 1x64x64x32, dim_alignments = 1x1x32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, undef, l1, sharded>>
        -> tensor<1x64x64x32xf32>
      d2m.yield %9 : (tensor<1x64x64x32xf32>)
    } : tensor<1x1x1x1x1x64x64x32xf32, #ttcore.metal_layout<logical_shape = 1x64x64x32, dim_alignments = 1x1x32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, undef, l1, sharded>>
    %7 = d2m.empty() : tensor<1x64x64x32xf32>
    %8 = d2m.to_layout %6, %7 : tensor<1x1x1x1x1x64x64x32xf32, #ttcore.metal_layout<logical_shape = 1x64x64x32, dim_alignments = 1x1x32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, undef, l1, sharded>> into tensor<1x64x64x32xf32> -> tensor<1x64x64x32xf32>
    return %8 : tensor<1x64x64x32xf32>
  }
}
