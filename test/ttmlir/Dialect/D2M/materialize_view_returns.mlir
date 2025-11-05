// RUN: ttmlir-opt --d2m-materialize-view-returns %s | FileCheck %s
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#layout1x1 = #ttcore.metal_layout<logical_shape = 256x768, dim_alignments = 32x256, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1>
#layout8x8 = #ttcore.metal_layout<logical_shape = 256x768, dim_alignments = 32x256, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1>

// Test basic case: to_layout then view, returning unmaterialized view
// This is the core problem - the view has no generic consumer
// CHECK-LABEL: @basic_to_layout_view_return
func.func @basic_to_layout_view_return(%arg0: tensor<256x768xf32>) -> tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout8x8> {
  // First do a to_layout to get onto device in 1x1 grid
  // CHECK: d2m.empty() : tensor<1x1x8x24x!ttcore.tile<32x32, f32>
  %empty_1x1 = d2m.empty() : tensor<1x1x8x24x!ttcore.tile<32x32, f32>, #layout1x1>

  // CHECK: d2m.to_layout
  %to_layout = d2m.to_layout %arg0, %empty_1x1 : tensor<256x768xf32> into tensor<1x1x8x24x!ttcore.tile<32x32, f32>, #layout1x1>
    -> tensor<1x1x8x24x!ttcore.tile<32x32, f32>, #layout1x1>

  // Then apply a view to redistribute to 8x8 grid
  // CHECK: %[[VIEW:.*]] = d2m.view_layout
  %view = d2m.view_layout %to_layout : tensor<1x1x8x24x!ttcore.tile<32x32, f32>, #layout1x1>
      -> tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout8x8>

  // CHECK: d2m.empty() : tensor<8x8x1x3x!ttcore.tile<32x32, f32>
  // CHECK: %[[MATERIALIZED:.*]] = d2m.generic
  // CHECK-SAME: grid = #ttcore.grid<8x8>
  // CHECK-SAME: threads = [#d2m.thread<datamovement>]
  // CHECK: ins(%[[VIEW]]
  // CHECK: d2m.reserve
  // CHECK: d2m.yield
  // CHECK: return %[[MATERIALIZED]]
  return %view : tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout8x8>
}

// Test that a view result returned directly gets materialized
// CHECK-LABEL: @view_returned_directly
func.func @view_returned_directly(%arg0: tensor<1x1x8x24x!ttcore.tile<32x32, f32>, #layout1x1>) -> tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout8x8> {
  // CHECK: %[[VIEW:.*]] = d2m.view_layout %arg0
  %view = d2m.view_layout %arg0 : tensor<1x1x8x24x!ttcore.tile<32x32, f32>, #layout1x1> -> tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout8x8>

  // CHECK: d2m.empty() : tensor<8x8x1x3x!ttcore.tile<32x32, f32>
  // CHECK: %[[MATERIALIZED:.*]] = d2m.generic
  // CHECK-SAME: grid = #ttcore.grid<8x8>
  // CHECK-SAME: threads = [#d2m.thread<datamovement>]
  // CHECK: ins(%[[VIEW]]
  // CHECK: d2m.reserve
  // CHECK: d2m.yield
  // CHECK: return %[[MATERIALIZED]]
  return %view : tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout8x8>
}

// Test that non-view returns are not affected
// CHECK-LABEL: @non_view_returned
func.func @non_view_returned(%arg0: tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout8x8>) -> tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout8x8> {
  // CHECK: return %arg0
  return %arg0 : tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout8x8>
}

// Test that views consumed by generic ops are not re-materialized
// CHECK-LABEL: @view_already_consumed
func.func @view_already_consumed(%arg0: tensor<1x1x8x24x!ttcore.tile<32x32, f32>, #layout1x1>) -> tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout8x8> {
  // CHECK: %[[VIEW:.*]] = d2m.view_layout %arg0
  %view = d2m.view_layout %arg0 : tensor<1x1x8x24x!ttcore.tile<32x32, f32>, #layout1x1> -> tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout8x8>
  %empty = d2m.empty() : tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout8x8>

  // CHECK: %[[RESULT:.*]] = d2m.generic
  // CHECK: ins(%[[VIEW]]
  %result = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<8x8>,
                         indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
                         iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>],
                         threads = [#d2m.thread<datamovement>]}
      ins(%view : tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout8x8>)
      outs(%empty : tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout8x8>) {
    ^bb0(%in_cb: !d2m.cb<tensor<1x3x!ttcore.tile<32x32, f32>>>,
         %out_cb: !d2m.cb<tensor<1x3x!ttcore.tile<32x32, f32>>>):
      %in = d2m.wait %in_cb : !d2m.cb<tensor<1x3x!ttcore.tile<32x32, f32>>> -> tensor<1x3x!ttcore.tile<32x32, f32>>
      %out = d2m.reserve %out_cb : !d2m.cb<tensor<1x3x!ttcore.tile<32x32, f32>>> -> tensor<1x3x!ttcore.tile<32x32, f32>>
      d2m.yield %out : (tensor<1x3x!ttcore.tile<32x32, f32>>)
  } : tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout8x8>

  // Verify the pass doesn't add another generic after this one
  // CHECK-NOT: d2m.generic
  // CHECK: return %[[RESULT]]
  return %result : tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout8x8>
}

// Test multiple returns with mixed view and non-view values
// CHECK-LABEL: @mixed_returns
func.func @mixed_returns(%arg0: tensor<1x1x8x24x!ttcore.tile<32x32, f32>, #layout1x1>, %arg1: tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout8x8>)
    -> (tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout8x8>, tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout8x8>) {
  // CHECK: %[[VIEW:.*]] = d2m.view_layout %arg0
  %view = d2m.view_layout %arg0 : tensor<1x1x8x24x!ttcore.tile<32x32, f32>, #layout1x1> -> tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout8x8>

  // CHECK: d2m.empty() : tensor<8x8x1x3x!ttcore.tile<32x32, f32>
  // CHECK: %[[MATERIALIZED:.*]] = d2m.generic
  // CHECK: ins(%[[VIEW]]
  // CHECK: d2m.reserve
  // CHECK: d2m.yield
  // CHECK: return %[[MATERIALIZED]], %arg1
  return %view, %arg1 : tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout8x8>, tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout8x8>
}
