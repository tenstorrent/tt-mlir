#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
module {
  module {
    ttcore.global @global_no_index = tensor<2x2xf32>
    func.func @test_missing_index(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
      %0 = d2m.empty() : tensor<2x2xf32>
      %1 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<compute>]}
          ins(%arg0 : tensor<2x2xf32>)
          outs(%0 : tensor<2x2xf32>)  {
      ^compute0(%cb0: !d2m.cb<tensor<2x2xf32>>, %cb1: !d2m.cb<tensor<2x2xf32>>):
        %2 = d2m.wait %cb0 : <tensor<2x2xf32>> -> tensor<2x2xf32>
        %3 = d2m.reserve %cb1 : <tensor<2x2xf32>> -> tensor<2x2xf32>
        d2m.yield
      } : tensor<2x2xf32>
      return %1 : tensor<2x2xf32>
    }
  }
  module {
    ttcore.global @global_with_index = tensor<2x2xf32> [0]
    func.func @test_nonexistent_global(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
      %0 = d2m.empty() : tensor<2x2xf32>
      %1 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<compute>]}
          ins(%arg0 : tensor<2x2xf32>)
          outs(%0 : tensor<2x2xf32>)  {
      ^compute0(%cb0: !d2m.cb<tensor<2x2xf32>>, %cb1: !d2m.cb<tensor<2x2xf32>>):
        %2 = d2m.wait %cb0 : <tensor<2x2xf32>> -> tensor<2x2xf32>
        %3 = d2m.reserve %cb1 : <tensor<2x2xf32>> -> tensor<2x2xf32>
        d2m.yield
      } : tensor<2x2xf32>
      return %1 : tensor<2x2xf32>
    }
  }
}
