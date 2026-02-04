// RUN: ttmlir-opt --ttcore-register-device --d2m-materialize-view-returns %s | FileCheck %s
// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#layout1x1 = #ttcore.metal_layout<logical_shape = 256x768, dim_alignments = 32x256, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded, index_map = map(0)>
#layout8x8 = #ttcore.metal_layout<logical_shape = 256x768, dim_alignments = 32x256, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded, index_map = map(0)>

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

  // Materialization uses load+store pair for proper CB association
  // CHECK: d2m.empty() : tensor<8x8x1x3x!ttcore.tile<32x32, f32>
  // CHECK: %[[MATERIALIZED:.*]] = d2m.generic
  // CHECK-SAME: grid = #ttcore.grid<8x8>
  // CHECK-SAME: threads = [#d2m.thread<unified>]
  // CHECK: ins(%[[VIEW]]
  // CHECK: d2m.remote_load
  // CHECK: d2m.remote_store
  // CHECK: d2m.yield
  // CHECK: return %[[MATERIALIZED]]
  return %view : tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout8x8>
}

// Test that a view result returned directly gets materialized
// CHECK-LABEL: @view_returned_directly
func.func @view_returned_directly(%arg0: tensor<1x1x8x24x!ttcore.tile<32x32, f32>, #layout1x1>) -> tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout8x8> {
  // CHECK: %[[VIEW:.*]] = d2m.view_layout %arg0
  %view = d2m.view_layout %arg0 : tensor<1x1x8x24x!ttcore.tile<32x32, f32>, #layout1x1> -> tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout8x8>

  // Materialization uses load+store pair for proper CB association
  // CHECK: d2m.empty() : tensor<8x8x1x3x!ttcore.tile<32x32, f32>
  // CHECK: %[[MATERIALIZED:.*]] = d2m.generic
  // CHECK-SAME: grid = #ttcore.grid<8x8>
  // CHECK-SAME: threads = [#d2m.thread<unified>]
  // CHECK: ins(%[[VIEW]]
  // CHECK: d2m.remote_load
  // CHECK: d2m.remote_store
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

  // Materialization uses load+store pair
  // CHECK: d2m.empty() : tensor<8x8x1x3x!ttcore.tile<32x32, f32>
  // CHECK: %[[MATERIALIZED:.*]] = d2m.generic
  // CHECK: ins(%[[VIEW]]
  // CHECK: d2m.remote_load
  // CHECK: d2m.remote_store
  // CHECK: d2m.yield
  // CHECK: return %[[MATERIALIZED]], %arg1
  return %view, %arg1 : tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout8x8>, tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout8x8>
}

// Test higher-rank tensors (3D uncollapsed) for permute TM use case
// Grid has leading 1 (1x4x4 -> 1x8x2), demonstrating higher rank with legal grid
// CHECK-LABEL: @higher_rank_view_return
#layout_6d_1x4x4 = #ttcore.metal_layout<logical_shape = 64x96x192, dim_alignments = 32x32x32, collapsed_intervals = dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>, undef, l1, sharded, index_map = map(0)>
#layout_6d_1x8x2 = #ttcore.metal_layout<logical_shape = 64x96x192, dim_alignments = 32x32x32, collapsed_intervals = dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>, undef, l1, sharded, index_map = map(0)>
func.func @higher_rank_view_return(%arg0: tensor<1x4x4x2x3x6x!ttcore.tile<32x32, f32>, #layout_6d_1x4x4>) -> tensor<1x8x2x1x6x6x!ttcore.tile<32x32, f32>, #layout_6d_1x8x2> {
  // CHECK: %[[VIEW:.*]] = d2m.view_layout %arg0
  %view = d2m.view_layout %arg0 : tensor<1x4x4x2x3x6x!ttcore.tile<32x32, f32>, #layout_6d_1x4x4> -> tensor<1x8x2x1x6x6x!ttcore.tile<32x32, f32>, #layout_6d_1x8x2>

  // Materialization uses load+store pair
  // CHECK: d2m.empty() : tensor<1x8x2x1x6x6x!ttcore.tile<32x32, f32>
  // CHECK: %[[MATERIALIZED:.*]] = d2m.generic
  // CHECK-SAME: grid = #ttcore.grid<1x8x2>
  // CHECK-SAME: threads = [#d2m.thread<unified>]
  // CHECK: ins(%[[VIEW]]
  // CHECK: d2m.remote_load
  // CHECK: d2m.remote_store
  // CHECK: d2m.yield
  // CHECK: return %[[MATERIALIZED]]
  return %view : tensor<1x8x2x1x6x6x!ttcore.tile<32x32, f32>, #layout_6d_1x8x2>
}

// Test Case 2: view consumed by device-to-host to_host before return
// Pattern: %view = view_layout ... -> %host = to_host %view -> return %host
// The view must be materialized BEFORE the device-to-host transfer
// CHECK-LABEL: @view_before_device_to_host
func.func @view_before_device_to_host(%arg0: tensor<1x1x8x24x!ttcore.tile<32x32, f32>, #layout1x1>) -> tensor<256x768xf32> {
  // Create a view that redistributes to 8x8 grid
  // CHECK: %[[VIEW:.*]] = d2m.view_layout %arg0
  %view = d2m.view_layout %arg0 : tensor<1x1x8x24x!ttcore.tile<32x32, f32>, #layout1x1> -> tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout8x8>

  // Allocate host output tensor for device-to-host transfer
  %host_empty = d2m.empty() : tensor<256x768xf32>

  // The view is consumed by a device-to-host to_host op.
  // The pass should materialize the view BEFORE this to_host op.
  // Materialization uses load+store pair
  // CHECK: d2m.empty() : tensor<8x8x1x3x!ttcore.tile<32x32, f32>
  // CHECK: %[[MATERIALIZED:.*]] = d2m.generic
  // CHECK-SAME: grid = #ttcore.grid<8x8>
  // CHECK-SAME: threads = [#d2m.thread<unified>]
  // CHECK: ins(%[[VIEW]]
  // CHECK: d2m.remote_load
  // CHECK: d2m.remote_store
  // CHECK: d2m.yield
  // CHECK: d2m.to_host %[[MATERIALIZED]]
  %to_host = d2m.to_host %view, %host_empty layout = #layout8x8 : tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout8x8> into tensor<256x768xf32> -> tensor<256x768xf32>

  // CHECK: return
  return %to_host : tensor<256x768xf32>
}

// Test that view before device-to-device to_layout is not affected (Case 2 should not apply)
// Only to_host ops trigger view materialization, not device-to-device to_layout ops.
// CHECK-LABEL: @view_before_device_to_device_unchanged
func.func @view_before_device_to_device_unchanged(%arg0: tensor<1x1x8x24x!ttcore.tile<32x32, f32>, #layout1x1>) -> tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout8x8> {
  // CHECK: %[[VIEW:.*]] = d2m.view_layout %arg0
  %view = d2m.view_layout %arg0 : tensor<1x1x8x24x!ttcore.tile<32x32, f32>, #layout1x1> -> tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout8x8>

  // Allocate device output tensor - this makes to_layout a device-to-device op
  %device_empty = d2m.empty() : tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout8x8>

  // This is a device-to-device to_layout (both have MetalLayoutAttr).
  // Case 2 only applies to to_host ops, not device-to-device to_layout.
  // CHECK: d2m.to_layout %[[VIEW]]
  %to_device = d2m.to_layout %view, %device_empty : tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout8x8> into tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout8x8> -> tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout8x8>

  // The result from a device-to-device to_layout is NOT a view, so no materialization needed
  // CHECK-NOT: d2m.generic
  // CHECK: return %{{.*}}
  return %to_device : tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout8x8>
}
