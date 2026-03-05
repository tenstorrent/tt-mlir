// RUN: ttmlir-opt --ttcore-register-device --d2m-lower-to-layout -o %t %s
// RUN: FileCheck %s --input-file=%t

#layout_src = #ttcore.metal_layout<logical_shape = 64x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>
#layout_dst = #ttcore.metal_layout<logical_shape = 64x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>

// Test remote input case: remote input, local output - should use remote_load
// CHECK-LABEL: func.func @test_remote_input
func.func @test_remote_input(%arg0: tensor<2x4x32x32xf32, #layout_src>, %arg1: tensor<4x2x32x32xf32, #layout_dst>) -> tensor<4x2x32x32xf32, #layout_dst> {
  %storage = d2m.empty() : tensor<2x4x32x32xf32, #layout_src>
  %stream = "d2m.stream_layout"(%arg0, %storage) <{remapping = #map4}> : (tensor<2x4x32x32xf32, #layout_src>, tensor<2x4x32x32xf32, #layout_src>) -> tensor<2x4x32x32xf32, #layout_src>

  // CHECK: %[[VIEW:.*]] = d2m.view_layout
  // CHECK: d2m.generic
  // CHECK-SAME: threads = [#d2m.thread<unified>]
  // CHECK: ^unified0
  // CHECK: %[[ITER0:.*]] = d2m.block_index(0) : index
  // CHECK: %[[ITER1:.*]] = d2m.block_index(1) : index
  // CHECK: d2m.remote_load %{{.*}} %[[VIEW]][%[[ITER0]], %[[ITER1]]]
  // CHECK: d2m.remote_store
  // CHECK-NOT: d2m.dma
  // CHECK: d2m.yield

  %1 = d2m.to_layout %stream, %arg1 : tensor<2x4x32x32xf32, #layout_src> into tensor<4x2x32x32xf32, #layout_dst>
    -> tensor<4x2x32x32xf32, #layout_dst>

  return %1 : tensor<4x2x32x32xf32, #layout_dst>
}
