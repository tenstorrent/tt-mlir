#layout = #ttcore.metal_layout<logical_shape = 1x64x64x32, dim_alignments = 1x1x32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, undef, l1,, index_map = map(0)>
#layout1 = #ttcore.metal_layout<logical_shape = 1x64x64x32, dim_alignments = 1x1x32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, undef, l1,, index_map = (d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d2, d1, d3, d4, d6, d5, d7)>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#parallel = #ttcore.iterator_type<parallel>
module attributes {ttcore.device = #ttcore.device<workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1] -> (0, 0, 0, d0 * s1 + d1 * s1 + d2 + s0), meshShape = , chipIds = [0]>} {
  func.func @test_grid_selection_row_major(%arg0: tensor<1x64x64x32xf32>) -> tensor<1x64x64x32xf32> {
    %0 = d2m.empty() : tensor<1x64x64x32xf32>
    %1 = d2m.empty() : tensor<1x1x1x1x1x64x64x32xf32, #layout>
    %2 = d2m.to_layout %arg0, %1 : tensor<1x64x64x32xf32> into tensor<1x1x1x1x1x64x64x32xf32, #layout> -> tensor<1x1x1x1x1x64x64x32xf32, #layout>
    %3 = d2m.empty() : tensor<1x1x1x1x1x64x64x32xf32, #layout>
    %4 = d2m.empty() : tensor<1x1x1x1x1x64x64x32xf32, #layout>
    %5 = d2m.empty() : tensor<1x1x1x1x1x64x64x32xf32, #layout1>
    %stream = "d2m.stream_layout"(%2, %5) : (tensor<1x1x1x1x1x64x64x32xf32, #layout>, tensor<1x1x1x1x1x64x64x32xf32, #layout1>) -> tensor<1x1x1x1x1x64x64x32xf32, #layout1>
    %6 = d2m.generic {block_factors = [1, 1, 1, 1], grid = #ttcore.grid<1x1x1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel, #parallel, #parallel], threads = [#d2m.thread<datamovement>]}
        ins(%stream : tensor<1x1x1x1x1x64x64x32xf32, #layout1>)
        outs(%4 : tensor<1x1x1x1x1x64x64x32xf32, #layout>)  {
    ^datamovement0(%cb0: !d2m.cb<tensor<1x64x64x32xf32>>, %cb1: !d2m.cb<tensor<1x64x64x32xf32>>):
      %9 = d2m.reserve %cb1 : <tensor<1x64x64x32xf32>> -> tensor<1x64x64x32xf32>
      %tx = d2m.dma %stream<#map>, %9 : (tensor<1x1x1x1x1x64x64x32xf32, #layout1>, tensor<1x64x64x32xf32>) -> !d2m.mem_tx
      d2m.dma_wait %tx
      d2m.yield
    } : tensor<1x1x1x1x1x64x64x32xf32, #layout>
    %7 = d2m.empty() : tensor<1x64x64x32xf32>
    %8 = d2m.to_layout %6, %7 : tensor<1x1x1x1x1x64x64x32xf32, #layout> into tensor<1x64x64x32xf32> -> tensor<1x64x64x32xf32>
    return %8 : tensor<1x64x64x32xf32>
  }
}
