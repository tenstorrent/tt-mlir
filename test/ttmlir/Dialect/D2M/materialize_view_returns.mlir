#layout = #ttcore.metal_layout<logical_shape = 256x768, dim_alignments = 32x256, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1,, index_map = map(0)>
#layout1 = #ttcore.metal_layout<logical_shape = 64x96x192, dim_alignments = 32x32x32, collapsed_intervals = dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>, undef, l1,, index_map = map(0)>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
module {
  func.func @basic_to_layout_view_return(%arg0: tensor<256x768xf32>) -> tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout> {
    %0 = d2m.empty() : tensor<1x1x8x24x!ttcore.tile<32x32, f32>, #layout>
    %1 = d2m.to_layout %arg0, %0 : tensor<256x768xf32> into tensor<1x1x8x24x!ttcore.tile<32x32, f32>, #layout> -> tensor<1x1x8x24x!ttcore.tile<32x32, f32>, #layout>
    %view = d2m.view_layout %1 : tensor<1x1x8x24x!ttcore.tile<32x32, f32>, #layout> -> tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout>
    return %view : tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout>
  }
  func.func @view_returned_directly(%arg0: tensor<1x1x8x24x!ttcore.tile<32x32, f32>, #layout>) -> tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout> {
    %view = d2m.view_layout %arg0 : tensor<1x1x8x24x!ttcore.tile<32x32, f32>, #layout> -> tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout>
    return %view : tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout>
  }
  func.func @non_view_returned(%arg0: tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout>) -> tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout> {
    return %arg0 : tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout>
  }
  func.func @view_already_consumed(%arg0: tensor<1x1x8x24x!ttcore.tile<32x32, f32>, #layout>) -> tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout> {
    %view = d2m.view_layout %arg0 : tensor<1x1x8x24x!ttcore.tile<32x32, f32>, #layout> -> tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout>
    %0 = d2m.empty() : tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout>
    %1 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<8x8>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<datamovement>]}
        ins(%view : tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout>)
        outs(%0 : tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout>)  {
    ^datamovement0(%cb0: !d2m.cb<tensor<1x3x!ttcore.tile<32x32, f32>>>, %cb1: !d2m.cb<tensor<1x3x!ttcore.tile<32x32, f32>>>):
      %2 = d2m.wait %cb0 : <tensor<1x3x!ttcore.tile<32x32, f32>>> -> tensor<1x3x!ttcore.tile<32x32, f32>>
      %3 = d2m.reserve %cb1 : <tensor<1x3x!ttcore.tile<32x32, f32>>> -> tensor<1x3x!ttcore.tile<32x32, f32>>
      d2m.yield
    } : tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout>
    return %1 : tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout>
  }
  func.func @mixed_returns(%arg0: tensor<1x1x8x24x!ttcore.tile<32x32, f32>, #layout>, %arg1: tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout>) -> (tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout>, tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout>) {
    %view = d2m.view_layout %arg0 : tensor<1x1x8x24x!ttcore.tile<32x32, f32>, #layout> -> tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout>
    return %view, %arg1 : tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout>, tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout>
  }
  func.func @higher_rank_view_return(%arg0: tensor<1x4x4x2x3x6x!ttcore.tile<32x32, f32>, #layout1>) -> tensor<1x8x2x1x6x6x!ttcore.tile<32x32, f32>, #layout1> {
    %view = d2m.view_layout %arg0 : tensor<1x4x4x2x3x6x!ttcore.tile<32x32, f32>, #layout1> -> tensor<1x8x2x1x6x6x!ttcore.tile<32x32, f32>, #layout1>
    return %view : tensor<1x8x2x1x6x6x!ttcore.tile<32x32, f32>, #layout1>
  }
}
