// RUN: ttmlir-opt --canonicalize %s --split-input-file | FileCheck %s
//
// Verifies the d2m.generic canonicalization that retargets nested linalg.generic
// DPS init operands to a fresh d2m.empty when the init's defining op is anything
// other than a d2m.empty. This catches cases where upstream linalg passes have
// fused the d2m.generic's input into the linalg.generic's output (in-place),
// which the backend does not support.

#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
#l1 = #ttcore.memory_space<l1>

// The linalg.generic outs is the d2m.reserve result; the canonicalization
// should insert a fresh d2m.empty immediately before the linalg.generic and
// rewire the outs.
// CHECK-LABEL: func.func @nested_linalg_reserve_init_retargeted
func.func @nested_linalg_reserve_init_retargeted(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = d2m.empty() : tensor<64x128xf32>
  // CHECK: d2m.generic
  // CHECK: %[[RESERVE:.*]] = d2m.reserve
  // CHECK: %[[EMPTY:.*]] = d2m.empty() : tensor<2x4x!ttcore.tile<32x32, f32>, #l1>
  // CHECK-NEXT: linalg.generic
  // CHECK-SAME: outs(%[[EMPTY]] :
  // CHECK: d2m.store %[[RESERVE]]
  %1 = "d2m.generic"(%arg0, %arg1, %0) <{
      block_factors = [1, 1],
      grid = #ttcore.grid<1x1>,
      indexing_maps = [#map, #map, #map],
      iterator_types = [#parallel, #parallel],
      threads = [#d2m.thread<compute>],
      operandSegmentSizes = array<i32: 2, 1, 0>}> ({
  ^bb0:
    %cb0 = d2m.get_cb(0) : !d2m.cb<tensor<2x4x!ttcore.tile<32x32, f32>, #l1>>
    %cb1 = d2m.get_cb(1) : !d2m.cb<tensor<2x4x!ttcore.tile<32x32, f32>, #l1>>
    %cb2 = d2m.get_cb(2) : !d2m.cb<tensor<2x4x!ttcore.tile<32x32, f32>, #l1>>
    %a = d2m.wait %cb0 : <tensor<2x4x!ttcore.tile<32x32, f32>, #l1>> -> tensor<2x4x!ttcore.tile<32x32, f32>, #l1>
    %b = d2m.wait %cb1 : <tensor<2x4x!ttcore.tile<32x32, f32>, #l1>> -> tensor<2x4x!ttcore.tile<32x32, f32>, #l1>
    %out = d2m.reserve %cb2 : <tensor<2x4x!ttcore.tile<32x32, f32>, #l1>> -> tensor<2x4x!ttcore.tile<32x32, f32>, #l1>
    %result = linalg.generic {
        indexing_maps = [#map, #map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%a, %b : tensor<2x4x!ttcore.tile<32x32, f32>, #l1>, tensor<2x4x!ttcore.tile<32x32, f32>, #l1>)
        outs(%out : tensor<2x4x!ttcore.tile<32x32, f32>, #l1>) {
    ^bb1(%aa: !ttcore.tile<32x32, f32>, %bb: !ttcore.tile<32x32, f32>, %cc: !ttcore.tile<32x32, f32>):
      %s = "d2m.tile_add"(%aa, %bb) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
      linalg.yield %s : !ttcore.tile<32x32, f32>
    } -> tensor<2x4x!ttcore.tile<32x32, f32>, #l1>
    d2m.store %out, %result : tensor<2x4x!ttcore.tile<32x32, f32>, #l1>
    d2m.yield %0 : (tensor<64x128xf32>)
  }) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}

// -----

// The linalg.generic outs is the linalg.generic's own input (the d2m.wait
// result), modelling the in-place pattern an upstream pass may produce. The
// canonicalization should still insert a fresh d2m.empty for the outs.
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
#l1 = #ttcore.memory_space<l1>

// CHECK-LABEL: func.func @nested_linalg_inplace_input_retargeted
func.func @nested_linalg_inplace_input_retargeted(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = d2m.empty() : tensor<64x128xf32>
  // CHECK: d2m.generic
  // CHECK: %[[WAIT:.*]] = d2m.wait
  // CHECK: %[[EMPTY:.*]] = d2m.empty() : tensor<2x4x!ttcore.tile<32x32, f32>, #l1>
  // CHECK-NEXT: linalg.generic
  // CHECK-SAME: ins(%[[WAIT]] :
  // CHECK-SAME: outs(%[[EMPTY]] :
  %1 = "d2m.generic"(%arg0, %0) <{
      block_factors = [1, 1],
      grid = #ttcore.grid<1x1>,
      indexing_maps = [#map, #map],
      iterator_types = [#parallel, #parallel],
      threads = [#d2m.thread<compute>],
      operandSegmentSizes = array<i32: 1, 1, 0>}> ({
  ^bb0:
    %cb0 = d2m.get_cb(0) : !d2m.cb<tensor<2x4x!ttcore.tile<32x32, f32>, #l1>>
    %cb1 = d2m.get_cb(1) : !d2m.cb<tensor<2x4x!ttcore.tile<32x32, f32>, #l1>>
    %a = d2m.wait %cb0 : <tensor<2x4x!ttcore.tile<32x32, f32>, #l1>> -> tensor<2x4x!ttcore.tile<32x32, f32>, #l1>
    %out = d2m.reserve %cb1 : <tensor<2x4x!ttcore.tile<32x32, f32>, #l1>> -> tensor<2x4x!ttcore.tile<32x32, f32>, #l1>
    %result = linalg.generic {
        indexing_maps = [#map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%a : tensor<2x4x!ttcore.tile<32x32, f32>, #l1>)
        outs(%a : tensor<2x4x!ttcore.tile<32x32, f32>, #l1>) {
    ^bb1(%aa: !ttcore.tile<32x32, f32>, %cc: !ttcore.tile<32x32, f32>):
      %s = "d2m.tile_abs"(%aa) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
      linalg.yield %s : !ttcore.tile<32x32, f32>
    } -> tensor<2x4x!ttcore.tile<32x32, f32>, #l1>
    d2m.store %out, %result : tensor<2x4x!ttcore.tile<32x32, f32>, #l1>
    d2m.yield %0 : (tensor<64x128xf32>)
  }) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}

// -----

// Negative: when the linalg.generic outs is already a d2m.empty, the
// canonicalization must not insert a redundant one.
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
#l1 = #ttcore.memory_space<l1>

// CHECK-LABEL: func.func @nested_linalg_empty_init_unchanged
func.func @nested_linalg_empty_init_unchanged(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = d2m.empty() : tensor<64x128xf32>
  // CHECK: d2m.generic
  // CHECK: %[[EMPTY:.*]] = d2m.empty() : tensor<2x4x!ttcore.tile<32x32, f32>, #l1>
  // CHECK-NOT: d2m.empty() : tensor<2x4x!ttcore.tile<32x32, f32>, #l1>
  // CHECK: linalg.generic
  // CHECK-SAME: outs(%[[EMPTY]] :
  %1 = "d2m.generic"(%arg0, %0) <{
      block_factors = [1, 1],
      grid = #ttcore.grid<1x1>,
      indexing_maps = [#map, #map],
      iterator_types = [#parallel, #parallel],
      threads = [#d2m.thread<compute>],
      operandSegmentSizes = array<i32: 1, 1, 0>}> ({
  ^bb0:
    %cb0 = d2m.get_cb(0) : !d2m.cb<tensor<2x4x!ttcore.tile<32x32, f32>, #l1>>
    %cb1 = d2m.get_cb(1) : !d2m.cb<tensor<2x4x!ttcore.tile<32x32, f32>, #l1>>
    %a = d2m.wait %cb0 : <tensor<2x4x!ttcore.tile<32x32, f32>, #l1>> -> tensor<2x4x!ttcore.tile<32x32, f32>, #l1>
    %out = d2m.reserve %cb1 : <tensor<2x4x!ttcore.tile<32x32, f32>, #l1>> -> tensor<2x4x!ttcore.tile<32x32, f32>, #l1>
    %scratch = d2m.empty() : tensor<2x4x!ttcore.tile<32x32, f32>, #l1>
    %result = linalg.generic {
        indexing_maps = [#map, #map],
        iterator_types = ["parallel", "parallel"]}
        ins(%a : tensor<2x4x!ttcore.tile<32x32, f32>, #l1>)
        outs(%scratch : tensor<2x4x!ttcore.tile<32x32, f32>, #l1>) {
    ^bb1(%aa: !ttcore.tile<32x32, f32>, %cc: !ttcore.tile<32x32, f32>):
      %s = "d2m.tile_abs"(%aa) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
      linalg.yield %s : !ttcore.tile<32x32, f32>
    } -> tensor<2x4x!ttcore.tile<32x32, f32>, #l1>
    d2m.store %out, %result : tensor<2x4x!ttcore.tile<32x32, f32>, #l1>
    d2m.yield %0 : (tensor<64x128xf32>)
  }) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}
