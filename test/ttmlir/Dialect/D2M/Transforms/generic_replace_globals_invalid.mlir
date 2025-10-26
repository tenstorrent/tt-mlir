// RUN: ttmlir-opt %s --d2m-generic-replace-globals --split-input-file -verify-diagnostics
#l1_ = #ttcore.memory_space<l1>
module {
  // Test case 1: Global without index attribute
  ttcore.global @global_no_index = tensor<2x2xf32>
  func.func @test_missing_index(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = d2m.empty() : tensor<2x2xf32>
    %1 = d2m.generic {
      block_factors = [1, 1],
      grid = #ttcore.grid<1x1>,
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>],
      threads = [#d2m.thread<compute>]}
      ins(%arg0 : tensor<2x2xf32>)
      outs(%0 : tensor<2x2xf32>) {
    ^bb0(%cb0: !d2m.cb<tensor<2x2xf32>>, %cb1: !d2m.cb<tensor<2x2xf32>>):
      %arg0_val = d2m.wait %cb0 : !d2m.cb<tensor<2x2xf32>> -> tensor<2x2xf32>
      %arg1_val = d2m.reserve %cb1 : !d2m.cb<tensor<2x2xf32>> -> tensor<2x2xf32>
      // expected-error@+2 {{Global must have a valid index attribute}}
      // expected-error@+1 {{failed to legalize operation 'ttcore.get_global'}}
      %global = ttcore.get_global @global_no_index : tensor<2x2xf32>
      d2m.yield %arg1_val : (tensor<2x2xf32>)
    } : tensor<2x2xf32>
    return %1 : tensor<2x2xf32>
  }
}
// -----
module {
  // Test case 2: Global with index but referencing non-existent global
  ttcore.global @global_with_index = tensor<2x2xf32> [0]
  func.func @test_nonexistent_global(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = d2m.empty() : tensor<2x2xf32>
    %1 = d2m.generic {
      block_factors = [1, 1],
      grid = #ttcore.grid<1x1>,
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>],
      threads = [#d2m.thread<compute>]}
      ins(%arg0 : tensor<2x2xf32>)
      outs(%0 : tensor<2x2xf32>) {
    ^bb0(%cb0: !d2m.cb<tensor<2x2xf32>>, %cb1: !d2m.cb<tensor<2x2xf32>>):
      %arg0_val = d2m.wait %cb0 : !d2m.cb<tensor<2x2xf32>> -> tensor<2x2xf32>
      %arg1_val = d2m.reserve %cb1 : !d2m.cb<tensor<2x2xf32>> -> tensor<2x2xf32>
      // expected-error@+2 {{Global symbol not found: "nonexistent_global"}}
      // expected-error@+1 {{failed to legalize operation 'ttcore.get_global'}}
      %global = ttcore.get_global @nonexistent_global : tensor<2x2xf32>
      d2m.yield %arg1_val : (tensor<2x2xf32>)
    } : tensor<2x2xf32>
    return %1 : tensor<2x2xf32>
  }
}
