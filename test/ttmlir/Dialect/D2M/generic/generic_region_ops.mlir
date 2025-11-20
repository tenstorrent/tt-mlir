#l1 = #ttcore.memory_space<l1>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
module {
  func.func @reduce_max(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = d2m.empty() : tensor<64x128xf32>
    %1 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<compute>]}
        ins(%arg0, %arg1 : tensor<64x128xf32>, tensor<64x128xf32>)
        outs(%0 : tensor<64x128xf32>)  {
    ^compute0(%cb0: !d2m.cb<tensor<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<tensor<2x4x!ttcore.tile<32x32, f32>, #l1>>, %cb2: !d2m.cb<tensor<2x4x!ttcore.tile<32x32, f32>, #l1>>):
      %2 = d2m.wait %cb0 : <tensor<2x4x!ttcore.tile<32x32, f32>, #l1>> -> tensor<2x4x!ttcore.tile<32x32, f32>, #l1>
      %3 = d2m.wait %cb1 : <tensor<2x4x!ttcore.tile<32x32, f32>, #l1>> -> tensor<2x4x!ttcore.tile<32x32, f32>, #l1>
      %4 = d2m.reserve %cb2 : <tensor<2x4x!ttcore.tile<32x32, f32>, #l1>> -> tensor<2x4x!ttcore.tile<32x32, f32>, #l1>
      %5 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%2, %3 : tensor<2x4x!ttcore.tile<32x32, f32>, #l1>, tensor<2x4x!ttcore.tile<32x32, f32>, #l1>) outs(%4 : tensor<2x4x!ttcore.tile<32x32, f32>, #l1>) {
      ^bb0(%in: !ttcore.tile<32x32, f32>, %in_0: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
        %6 = "d2m.tile_reduce_max"(%in, %in_0, %out) <{reduce_dim = #d2m<reduce_dim R>}> : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %6 : !ttcore.tile<32x32, f32>
      } -> tensor<2x4x!ttcore.tile<32x32, f32>, #l1>
      d2m.yield
    } : tensor<64x128xf32>
    return %1 : tensor<64x128xf32>
  }
}
