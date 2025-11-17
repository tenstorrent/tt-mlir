// RUN: ttmlir-opt --canonicalize %s --split-input-file | FileCheck %s
//
// This test verifies a canonicalization that eliminates d2m.empty() operations
// used as outputs in DPS operations (like linalg.generic) by replacing them
// with d2m.wait/reserve results from the same circular buffer argument.
//
// The test checks for this canonicalizer bug: When multiple wait/reserve
// operations exist on the same circular buffer, the canonicalization picks
// a definition that does not dominate the use, resulting in invalid IR.
//
// This test has two operations on the output circular buffer:
//   1. d2m.reserve (before linalg.generic) - correct choice
//   2. d2m.wait (after linalg.generic) - incorrect, causes domination error
// The test verifies that the canonicalization picks #1, not #2.

#layout = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1>
#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @canonicalize_with_multiple_reserves
func.func @canonicalize_with_multiple_reserves(%arg0: tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>) -> tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout> {
  %0 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>

  // CHECK: d2m.generic
  // CHECK: ^compute0(%[[CB_IN:.*]]: !d2m.cb<{{.*}}>, %[[CB_OUT:.*]]: !d2m.cb<{{.*}}>):
  // CHECK-NEXT: d2m.wait %[[CB_IN]]
  // CHECK-NEXT: %[[RESERVE:.*]] = d2m.reserve %[[CB_OUT]]
  // CHECK-NEXT: d2m.empty()
  // CHECK-NEXT: %[[RESULT:.*]] = linalg.generic
  // CHECK-SAME: outs(%[[RESERVE]] :
  // CHECK: d2m.store %[[RESERVE]], %[[RESULT]]
  // CHECK: %[[WAIT:.*]] = d2m.wait %[[CB_OUT]]
  // CHECK: d2m.yield %[[WAIT]]
  %1 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>], threads = [#d2m.thread<compute>]}
      ins(%arg0 : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>)
      outs(%0 : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>)  {
  ^compute0(%cb_in: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>, %cb_out: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>):
    %in = d2m.wait %cb_in : <tensor<1x1x!ttcore.tile<32x32, f32>>> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %out1 = d2m.reserve %cb_out : <tensor<1x1x!ttcore.tile<32x32, f32>>> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %temp = d2m.empty() : tensor<1x1x!ttcore.tile<32x32, f32>>
    %result = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
        ins(%in : tensor<1x1x!ttcore.tile<32x32, f32>>)
        outs(%temp : tensor<1x1x!ttcore.tile<32x32, f32>>) {
    ^bb0(%in_val: !ttcore.tile<32x32, f32>, %out_val: !ttcore.tile<32x32, f32>):
      %abs = "d2m.tile_abs"(%in_val) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
      linalg.yield %abs : !ttcore.tile<32x32, f32>
    } -> tensor<1x1x!ttcore.tile<32x32, f32>>
    d2m.store %out1, %result : tensor<1x1x!ttcore.tile<32x32, f32>>
    %out2 = d2m.wait %cb_out : <tensor<1x1x!ttcore.tile<32x32, f32>>> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    d2m.yield %out2 : (tensor<1x1x!ttcore.tile<32x32, f32>>)
  } : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>

  return %1 : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
}

// -----

// This test verifies that the canonicalization does NOT happen when there is
// no dominating wait/reserve operation for the output circular buffer.

#layout = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1>
#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @no_canonicalization_without_dominating_op
func.func @no_canonicalization_without_dominating_op(%arg0: tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>) -> tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout> {
  %0 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>

  // CHECK: d2m.generic
  // CHECK: ^compute0(%[[CB_IN:.*]]: !d2m.cb<{{.*}}>, %[[CB_OUT:.*]]: !d2m.cb<{{.*}}>):
  // CHECK-NEXT: d2m.wait %[[CB_IN]]
  // CHECK-NEXT: %[[EMPTY:.*]] = d2m.empty()
  // CHECK-NEXT: %[[RESULT:.*]] = linalg.generic
  // CHECK-SAME: outs(%[[EMPTY]] :
  // CHECK: %[[RESERVE:.*]] = d2m.reserve %[[CB_OUT]]
  // CHECK: d2m.store %[[RESERVE]], %[[RESULT]]
  // CHECK: d2m.yield
  %1 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>], threads = [#d2m.thread<compute>]}
      ins(%arg0 : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>)
      outs(%0 : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>)  {
  ^compute0(%cb_in: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>, %cb_out: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>):
    %in = d2m.wait %cb_in : <tensor<1x1x!ttcore.tile<32x32, f32>>> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %temp = d2m.empty() : tensor<1x1x!ttcore.tile<32x32, f32>>
    %result = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
        ins(%in : tensor<1x1x!ttcore.tile<32x32, f32>>)
        outs(%temp : tensor<1x1x!ttcore.tile<32x32, f32>>) {
    ^bb0(%in_val: !ttcore.tile<32x32, f32>, %out_val: !ttcore.tile<32x32, f32>):
      %abs = "d2m.tile_abs"(%in_val) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
      linalg.yield %abs : !ttcore.tile<32x32, f32>
    } -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %out = d2m.reserve %cb_out : <tensor<1x1x!ttcore.tile<32x32, f32>>> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    d2m.store %out, %result : tensor<1x1x!ttcore.tile<32x32, f32>>
    d2m.yield %out : (tensor<1x1x!ttcore.tile<32x32, f32>>)
  } : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>

  return %1 : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
}

// -----

// Test that exercises the while loop that walks up the parent chain to find
// a FuncOp. When d2m.generic is nested inside a loop (like scf.for), the
// while loop at line 1270 executes, covering line 1271.

#layout = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1>
#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @test_nested_in_loop
func.func @test_nested_in_loop(%arg0: tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>) -> tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout> {
  %0 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index

  // CHECK: scf.for
  %1 = scf.for %i = %c0 to %c2 step %c1 iter_args(%iter_arg = %0) -> (tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>) {
    // d2m.generic nested inside scf.for - exercises parent chain walk
    // CHECK: d2m.generic
    // CHECK: ^compute0(%[[CB_IN:.*]]: !d2m.cb<{{.*}}>, %[[CB_OUT:.*]]: !d2m.cb<{{.*}}>):
    // CHECK-NEXT: d2m.wait %[[CB_IN]]
    // CHECK-NEXT: %[[RESERVE:.*]] = d2m.reserve %[[CB_OUT]]
    // CHECK-NEXT: d2m.empty()
    // CHECK-NEXT: linalg.generic
    // CHECK-SAME: outs(%[[RESERVE]] :
    %2 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>], threads = [#d2m.thread<compute>]}
        ins(%arg0 : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>)
        outs(%iter_arg : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>)  {
    ^compute0(%cb_in: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>, %cb_out: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>):
      %in = d2m.wait %cb_in : <tensor<1x1x!ttcore.tile<32x32, f32>>> -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %reserve = d2m.reserve %cb_out : <tensor<1x1x!ttcore.tile<32x32, f32>>> -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %temp = d2m.empty() : tensor<1x1x!ttcore.tile<32x32, f32>>
      %result = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
          ins(%in : tensor<1x1x!ttcore.tile<32x32, f32>>)
          outs(%temp : tensor<1x1x!ttcore.tile<32x32, f32>>) {
      ^bb0(%in_val: !ttcore.tile<32x32, f32>, %out_val: !ttcore.tile<32x32, f32>):
        %abs = "d2m.tile_abs"(%in_val) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %abs : !ttcore.tile<32x32, f32>
      } -> tensor<1x1x!ttcore.tile<32x32, f32>>
      d2m.yield %result : (tensor<1x1x!ttcore.tile<32x32, f32>>)
    } : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
    scf.yield %2 : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
  }
  return %1 : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
}

// -----

// This test verifies the canonicalization pattern uses DominanceInfo for
// cross-block dominance checking when replacing d2m.empty() operations with
// wait/reserve results from circular buffers. The test places the reserve
// operation in one block and the DPS operation (linalg.generic) in a different
// block to ensure dominance analysis is required rather than simple same-block
// ordering checks.

#layout = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1>
#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @canonicalize_dps_cross_block_dominance
func.func @canonicalize_dps_cross_block_dominance(%arg0: tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>) -> tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout> {
  %empty = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>

  // CHECK: d2m.generic
  // CHECK: ^compute0(%[[CB_IN:.*]]: !d2m.cb<{{.*}}>, %[[CB_OUT:.*]]: !d2m.cb<{{.*}}>):
  // Entry block: create wait/reserve before branching
  // CHECK-NEXT: d2m.wait %[[CB_IN]]
  // CHECK-NEXT: %[[RESERVE:.*]] = d2m.reserve %[[CB_OUT]]
  %result = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>], threads = [#d2m.thread<compute>]}
      ins(%arg0 : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>)
      outs(%empty : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>)  {
  ^compute0(%cb_in: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>, %cb_out: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>):
    %in = d2m.wait %cb_in : <tensor<1x1x!ttcore.tile<32x32, f32>>> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %reserve = d2m.reserve %cb_out : <tensor<1x1x!ttcore.tile<32x32, f32>>> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %temp = d2m.empty() : tensor<1x1x!ttcore.tile<32x32, f32>>

    // Create a new block (then branch) to host the DPS op, ensuring cross-block dominance is required.
    %ctrue = arith.constant true
    scf.if %ctrue {
      // CHECK: %[[GEN:.*]] = linalg.generic
      // CHECK-SAME: outs(%[[RESERVE]] :
      %gen = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
          ins(%in : tensor<1x1x!ttcore.tile<32x32, f32>>)
          outs(%temp : tensor<1x1x!ttcore.tile<32x32, f32>>) {
      ^bb0(%in_val: !ttcore.tile<32x32, f32>, %out_val: !ttcore.tile<32x32, f32>):
        %abs = "d2m.tile_abs"(%in_val) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %abs : !ttcore.tile<32x32, f32>
      } -> tensor<1x1x!ttcore.tile<32x32, f32>>
      // CHECK: d2m.store %[[RESERVE]], %[[GEN]]
      d2m.store %reserve, %gen : tensor<1x1x!ttcore.tile<32x32, f32>>
    }

    d2m.yield %reserve : (tensor<1x1x!ttcore.tile<32x32, f32>>)
  } : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>

  // CHECK: return
  return %result : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
}
