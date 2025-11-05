// RUN: ttmlir-opt %s --split-input-file --verify-diagnostics

#layout = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1>

// Verify that indexing_maps must be non-empty when not in explicit datamovement form.
// This test has non-empty block_factors but empty indexing_maps, which should fail.

func.func @test_empty_indexing_maps_with_block_factors(%arg0: tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>) -> tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout> {
  %0 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>

  // expected-error @+1 {{indexing_maps must be non-empty unless in explicit datamovement form}}
  %1 = d2m.generic {
    block_factors = [1, 1],
    grid = #ttcore.grid<1x1>,
    indexing_maps = [],
    iterator_types = [],
    threads = [#d2m.thread<compute>]
  }
  ins(%arg0 : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>)
  outs(%0 : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>)  {
  ^compute0(%cb_in: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>, %cb_out: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>):
    %in = d2m.wait %cb_in : <tensor<1x1x!ttcore.tile<32x32, f32>>> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %out = d2m.reserve %cb_out : <tensor<1x1x!ttcore.tile<32x32, f32>>> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    d2m.yield %out : (tensor<1x1x!ttcore.tile<32x32, f32>>)
  } : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>

  return %1 : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
}

// -----

#layout = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1>

// Verify that indexing_maps must be non-empty when iterator_types is non-empty.
// This test has non-empty iterator_types but empty indexing_maps, which should fail.

func.func @test_empty_indexing_maps_with_iterator_types(%arg0: tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>) -> tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout> {
  %0 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>

  // expected-error @+1 {{indexing_maps must be non-empty unless in explicit datamovement form}}
  %1 = d2m.generic {
    block_factors = [1, 1],
    grid = #ttcore.grid<1x1>,
    indexing_maps = [],
    iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>],
    threads = [#d2m.thread<compute>]
  }
  ins(%arg0 : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>)
  outs(%0 : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>)  {
  ^compute0(%cb_in: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>, %cb_out: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>):
    %in = d2m.wait %cb_in : <tensor<1x1x!ttcore.tile<32x32, f32>>> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %out = d2m.reserve %cb_out : <tensor<1x1x!ttcore.tile<32x32, f32>>> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    d2m.yield %out : (tensor<1x1x!ttcore.tile<32x32, f32>>)
  } : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>

  return %1 : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
}

// -----

#layout = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1>
#map = affine_map<(d0, d1) -> (d0, d1)>

// Verify that explicit datamovement form (all three empty) is allowed.

func.func @test_explicit_datamovement_form(%arg0: tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>) -> tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout> {
  %0 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>

  // This should be valid: all three attributes are empty
  %1 = d2m.generic {
    block_factors = [],
    grid = #ttcore.grid<1x1>,
    indexing_maps = [],
    iterator_types = [],
    threads = [#d2m.thread<compute>]
  }
  ins(%arg0 : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>)
  outs(%0 : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>)  {
  ^compute0(%cb_in: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>, %cb_out: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>):
    %in = d2m.wait %cb_in : <tensor<1x1x!ttcore.tile<32x32, f32>>> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %out = d2m.reserve %cb_out : <tensor<1x1x!ttcore.tile<32x32, f32>>> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    d2m.yield %out : (tensor<1x1x!ttcore.tile<32x32, f32>>)
  } : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>

  return %1 : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
}
