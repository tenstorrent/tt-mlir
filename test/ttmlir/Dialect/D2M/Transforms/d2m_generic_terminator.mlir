// RUN: ttmlir-opt -verify-diagnostics --split-input-file %s

// Test that d2m.generic requires a d2m.yield terminator

#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
#l1 = #ttcore.memory_space<l1>

module {
  func.func @test_wrong_yield_arguments() -> tensor<2x2xf32> {
    %0 = d2m.empty() : tensor<2x2xf32>
    %1 = d2m.empty() : tensor<2x2xf32>
    %2 = d2m.empty() : tensor<2x2xf32>
    // expected-error @+1 {{'d2m.generic' op yield terminator must have the same number of arguments as generic results}}
    %result = "d2m.generic"(%0, %1, %2) <{
        block_factors = [1, 1],
        grid = #ttcore.grid<1x1>,
        indexing_maps = [#map, #map, #map],
        iterator_types = [#parallel, #parallel],
        threads = [#d2m.thread<compute>],
        operandSegmentSizes = array<i32: 2, 1>
        }> ({
        ^bb0(%cb0: !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>, #l1>>,
             %cb1: !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>, #l1>>,
             %cb2: !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>, #l1>>):
          %arg0 = d2m.wait %cb0 : !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>, #l1>> -> tensor<2x2x!ttcore.tile<32x32, f32>, #l1>
          %arg1 = d2m.wait %cb1 : !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>, #l1>> -> tensor<2x2x!ttcore.tile<32x32, f32>, #l1>
          %arg2 = d2m.reserve %cb2 : !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>, #l1>> -> tensor<2x2x!ttcore.tile<32x32, f32>, #l1>
          // Yield with wrong number of arguments - should fail verifier
          d2m.yield %arg0 : (tensor<2x2x!ttcore.tile<32x32, f32>, #l1>)
        }) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    return %result : tensor<2x2xf32>
  }
}

// -----

#map2 = affine_map<(d0, d1) -> (d0, d1)>
#parallel2 = #ttcore.iterator_type<parallel>
#l1_2 = #ttcore.memory_space<l1>

module {
  func.func @test_correct_terminator() -> tensor<2x2xf32> {
    %0 = d2m.empty() : tensor<2x2xf32>
    %1 = d2m.empty() : tensor<2x2xf32>
    %2 = d2m.empty() : tensor<2x2xf32>
    %result = "d2m.generic"(%0, %1, %2) <{
        block_factors = [1, 1],
        grid = #ttcore.grid<1x1>,
        indexing_maps = [#map2, #map2, #map2],
        iterator_types = [#parallel2, #parallel2],
        threads = [#d2m.thread<compute>],
        operandSegmentSizes = array<i32: 2, 1>
        }> ({
        ^bb0(%cb0: !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>, #l1_2>>,
             %cb1: !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>, #l1_2>>,
             %cb2: !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>, #l1_2>>):
          %arg0 = d2m.wait %cb0 : !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>, #l1_2>> -> tensor<2x2x!ttcore.tile<32x32, f32>, #l1_2>
          %arg1 = d2m.wait %cb1 : !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>, #l1_2>> -> tensor<2x2x!ttcore.tile<32x32, f32>, #l1_2>
          %arg2 = d2m.reserve %cb2 : !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>, #l1_2>> -> tensor<2x2x!ttcore.tile<32x32, f32>, #l1_2>
          // Correct terminator
          d2m.yield
        }) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    return %result : tensor<2x2xf32>
  }
}
