#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
module {
  module {
    ttcore.global @global_0 = tensor<2x2xf32> [0]
    func.func @test_no_replacement_outside_generic() -> tensor<2x2xf32> {
      %0 = ttcore.get_global @global_0 : tensor<2x2xf32>
      return %0 : tensor<2x2xf32>
    }
  }
  module {
    ttcore.global @global_0 = tensor<2x2xf32> [0]
    ttcore.global @global_1 = tensor<2x2xf32> [1]
    func.func @test_replace_globals(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
      %0 = d2m.empty() : tensor<2x2xf32>
      %1 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<compute>]}
          ins(%arg0, %arg1 : tensor<2x2xf32>, tensor<2x2xf32>)
          outs(%0 : tensor<2x2xf32>)  {
      ^compute0(%cb0: !d2m.cb<tensor<2x2xf32>>, %cb1: !d2m.cb<tensor<2x2xf32>>, %cb2: !d2m.cb<tensor<2x2xf32>>):
        %2 = ttcore.get_global @global_0 : tensor<2x2xf32>
        %3 = ttcore.get_global @global_1 : tensor<2x2xf32>
        %4 = d2m.wait %cb0 : <tensor<2x2xf32>> -> tensor<2x2xf32>
        %5 = d2m.wait %cb1 : <tensor<2x2xf32>> -> tensor<2x2xf32>
        %6 = d2m.reserve %cb2 : <tensor<2x2xf32>> -> tensor<2x2xf32>
        %7 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%2, %3 : tensor<2x2xf32>, tensor<2x2xf32>) outs(%6 : tensor<2x2xf32>) {
        ^bb0(%in: f32, %in_0: f32, %out: f32):
          %8 = arith.addf %in, %in_0 : f32
          linalg.yield %8 : f32
        } -> tensor<2x2xf32>
        d2m.yield
      } : tensor<2x2xf32>
      return %1 : tensor<2x2xf32>
    }
  }
  module {
    ttcore.global @input_global = tensor<2x2xf32> [0]
    ttcore.global @weight_global = tensor<2x2xf32> [1]
    ttcore.global @bias_global = tensor<2x2xf32> [2]
    func.func @test_comprehensive_replacement(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>, %arg2: tensor<2x2xf32>) -> tensor<2x2xf32> {
      %0 = d2m.empty() : tensor<2x2xf32>
      %1 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map, #map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<compute>]}
          ins(%arg0, %arg1, %arg2 : tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>)
          outs(%0 : tensor<2x2xf32>)  {
      ^compute0(%cb0: !d2m.cb<tensor<2x2xf32>>, %cb1: !d2m.cb<tensor<2x2xf32>>, %cb2: !d2m.cb<tensor<2x2xf32>>, %cb3: !d2m.cb<tensor<2x2xf32>>):
        %c0 = arith.constant 0 : index
        %2 = ttcore.get_global @input_global : tensor<2x2xf32>
        %3 = ttcore.get_global @weight_global : tensor<2x2xf32>
        %4 = ttcore.get_global @bias_global : tensor<2x2xf32>
        %5 = d2m.wait %cb0 : <tensor<2x2xf32>> -> tensor<2x2xf32>
        %6 = d2m.wait %cb1 : <tensor<2x2xf32>> -> tensor<2x2xf32>
        %7 = d2m.wait %cb2 : <tensor<2x2xf32>> -> tensor<2x2xf32>
        %8 = d2m.reserve %cb3 : <tensor<2x2xf32>> -> tensor<2x2xf32>
        %extracted = tensor.extract %2[%c0, %c0] : tensor<2x2xf32>
        %extracted_0 = tensor.extract %3[%c0, %c0] : tensor<2x2xf32>
        %extracted_1 = tensor.extract %4[%c0, %c0] : tensor<2x2xf32>
        %9 = arith.mulf %extracted, %extracted_0 : f32
        %10 = arith.addf %9, %extracted_1 : f32
        %inserted = tensor.insert %10 into %8[%c0, %c0] : tensor<2x2xf32>
        d2m.yield
      } : tensor<2x2xf32>
      return %1 : tensor<2x2xf32>
    }
    func.func @test_mixed_usage(%arg0: tensor<2x2xf32>) -> (tensor<2x2xf32>, tensor<2x2xf32>) {
      %0 = ttcore.get_global @input_global : tensor<2x2xf32>
      %1 = d2m.empty() : tensor<2x2xf32>
      %2 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<compute>]}
          ins(%arg0 : tensor<2x2xf32>)
          outs(%1 : tensor<2x2xf32>)  {
      ^compute0(%cb0: !d2m.cb<tensor<2x2xf32>>, %cb1: !d2m.cb<tensor<2x2xf32>>):
        %c2 = arith.constant 2 : index
        %c1 = arith.constant 1 : index
        %c0 = arith.constant 0 : index
        %3 = ttcore.get_global @input_global : tensor<2x2xf32>
        %4 = d2m.wait %cb0 : <tensor<2x2xf32>> -> tensor<2x2xf32>
        %5 = d2m.reserve %cb1 : <tensor<2x2xf32>> -> tensor<2x2xf32>
        %6 = scf.for %arg1 = %c0 to %c2 step %c1 iter_args(%arg2 = %5) -> (tensor<2x2xf32>) {
          %extracted = tensor.extract %3[%arg1, %arg1] : tensor<2x2xf32>
          %inserted = tensor.insert %extracted into %arg2[%arg1, %arg1] : tensor<2x2xf32>
          scf.yield %inserted : tensor<2x2xf32>
        }
        d2m.yield
      } : tensor<2x2xf32>
      return %0, %2 : tensor<2x2xf32>, tensor<2x2xf32>
    }
  }
}
