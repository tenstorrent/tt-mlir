// RUN: ttmlir-opt --split-input-file --canonicalize %s | FileCheck %s

// Verify that d2m.generic operations in explicit datamovement form
// (empty block_factors, indexing_maps, iterator_types) can have multiple
// regions without yield terminators. In this form, users manage all data
// movement manually and provide custom terminators in each region.

#layout = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1>
#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @explicit_datamovement_multi_region
func.func @explicit_datamovement_multi_region(
  %arg0: tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>,
  %arg1: tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>,
  %arg2: tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>
) -> tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout> {
  %0 = "d2m.empty"() : () -> tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>
  %stream0 = "d2m.stream_layout"(%arg0, %0) : (tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>, tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>) -> tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>
  %1 = "d2m.empty"() : () -> tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>
  %stream1 = "d2m.stream_layout"(%arg1, %1) : (tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>, tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>) -> tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>

  // CHECK: d2m.generic
  // CHECK-SAME: block_factors = []
  // CHECK-SAME: indexing_maps = []
  // CHECK-SAME: iterator_types = []
  %result = "d2m.generic"(%stream0, %stream1, %arg2) <{
    block_factors = [],
    grid = #ttcore.grid<1x1>,
    indexing_maps = [],
    iterator_types = [],
    operandSegmentSizes = array<i32: 2, 1>,
    threads = [#d2m.thread<datamovement>, #d2m.thread<datamovement>, #d2m.thread<compute>]
  }> ({
  ^bb0(%cb0: !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>>>, %cb1: !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>>>, %cb2: !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>>>):
    %2 = "d2m.reserve"(%cb0) : (!d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>>>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
    %c0 = "arith.constant"() <{value = 0 : index}> : () -> index
  }, {
  ^bb0(%cb0: !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>>>, %cb1: !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>>>, %cb2: !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>>>):
    %2 = "d2m.reserve"(%cb1) : (!d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>>>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
    %c0 = "arith.constant"() <{value = 0 : index}> : () -> index
  }, {
  ^bb0(%cb0: !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>>>, %cb1: !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>>>, %cb2: !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>>>):
    %3 = "d2m.wait"(%cb0) : (!d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>>>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
    %4 = "d2m.wait"(%cb1) : (!d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>>>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
    %5 = "d2m.reserve"(%cb2) : (!d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>>>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
    %6 = "d2m.empty"() : () -> tensor<2x2x!ttcore.tile<32x32, f32>>
    %7 = "linalg.generic"(%3, %4, %6) <{
      indexing_maps = [#map, #map, #map],
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>],
      operandSegmentSizes = array<i32: 2, 1>
    }> ({
    ^bb0(%in: !ttcore.tile<32x32, f32>, %in_1: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
      %8 = "d2m.tile_add"(%in, %in_1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
      "linalg.yield"(%8) : (!ttcore.tile<32x32, f32>) -> ()
    }) : (tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
    "d2m.store"(%5, %7) : (tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>) -> ()
  }) : (tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>, tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>, tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>) -> tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>

  return %result : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout>
}
