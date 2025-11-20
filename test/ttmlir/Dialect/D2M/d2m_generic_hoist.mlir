#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
module {
  func.func @forward(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = d2m.empty() : tensor<64x128xf32>
    %1 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<compute>]}
        ins(%arg0, %arg1 : tensor<64x128xf32>, tensor<64x128xf32>)
        outs(%0 : tensor<64x128xf32>)  {
    ^compute0(%cb0: !d2m.cb<tensor<64x128xf32>>, %cb1: !d2m.cb<tensor<64x128xf32>>, %cb2: !d2m.cb<tensor<64x128xf32>>):
      %cst = arith.constant 0.000000e+00 : f32
      %c0 = arith.constant 0 : index
      %2 = d2m.wait %cb0 : <tensor<64x128xf32>> -> tensor<64x128xf32>
      %3 = d2m.wait %cb1 : <tensor<64x128xf32>> -> tensor<64x128xf32>
      %4 = d2m.reserve %cb2 : <tensor<64x128xf32>> -> tensor<64x128xf32>
      %extracted = tensor.extract %2[%c0, %c0] : tensor<64x128xf32>
      %5 = arith.addf %extracted, %cst : f32
      %inserted = tensor.insert %5 into %4[%c0, %c0] : tensor<64x128xf32>
      d2m.yield
    } : tensor<64x128xf32>
    return %1 : tensor<64x128xf32>
  }
}
