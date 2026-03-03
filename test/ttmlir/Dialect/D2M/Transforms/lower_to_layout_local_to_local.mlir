// RUN: ttmlir-opt --ttcore-register-device --d2m-lower-to-layout -o %t %s
// RUN: FileCheck %s --input-file=%t

#layout = #ttcore.metal_layout<logical_shape = 256x256, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>

// Test local-to-local transfer: both input and output are local (not remote)
// This should use remote_load instead of DMAOp
// CHECK-LABEL: func.func @test_local_to_local_reblock
func.func @test_local_to_local_reblock(%arg0: tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #layout>) -> tensor<4x4x2x2x!ttcore.tile<32x32, f32>, #layout> {
  %0 = d2m.empty() : tensor<4x4x2x2x!ttcore.tile<32x32, f32>, #layout>

  // CHECK: %[[VIEW:.*]] = d2m.view_layout
  // CHECK: d2m.generic
  // CHECK-SAME: threads = [#d2m.thread<unified>]
  // CHECK: d2m.block_index(0)
  // CHECK: d2m.block_index(1)
  // CHECK: d2m.remote_load %{{.*}} %[[VIEW]][%{{.*}}, %{{.*}}]
  // CHECK-NOT: d2m.dma
  // CHECK: d2m.yield

  %1 = d2m.to_layout %arg0, %0 : tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #layout> into tensor<4x4x2x2x!ttcore.tile<32x32, f32>, #layout>
    -> tensor<4x4x2x2x!ttcore.tile<32x32, f32>, #layout>

  return %1 : tensor<4x4x2x2x!ttcore.tile<32x32, f32>, #layout>
}

// Test the original bug case: multiple local-to-local reblocking operations
// CHECK-LABEL: func.func @test_multiple_local_reblocks
func.func @test_multiple_local_reblocks(%arg0: tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #layout>) -> tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #layout> {
  %0 = d2m.empty() : tensor<4x4x2x2x!ttcore.tile<32x32, f32>, #layout>
  %1 = d2m.empty() : tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #layout>

  // First reblock: 1x1x8x8 -> 4x4x2x2
  // CHECK: %[[VIEW1:.*]] = d2m.view_layout
  // CHECK: d2m.generic
  // CHECK-SAME: threads = [#d2m.thread<unified>]
  // CHECK: d2m.block_index(0)
  // CHECK: d2m.block_index(1)
  // CHECK: d2m.remote_load %{{.*}} %[[VIEW1]][%{{.*}}, %{{.*}}]
  // CHECK-NOT: d2m.dma

  %2 = d2m.to_layout %arg0, %0 : tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #layout> into tensor<4x4x2x2x!ttcore.tile<32x32, f32>, #layout>
    -> tensor<4x4x2x2x!ttcore.tile<32x32, f32>, #layout>

  // Second reblock: 4x4x2x2 -> 1x1x8x8
  // CHECK: %[[VIEW2:.*]] = d2m.view_layout
  // CHECK: d2m.generic
  // CHECK-SAME: threads = [#d2m.thread<unified>]
  // CHECK: d2m.block_index(0)
  // CHECK: d2m.block_index(1)
  // CHECK: d2m.remote_load %{{.*}} %[[VIEW2]][%{{.*}}, %{{.*}}]
  // CHECK-NOT: d2m.dma

  %3 = d2m.to_layout %2, %1 : tensor<4x4x2x2x!ttcore.tile<32x32, f32>, #layout> into tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #layout>
    -> tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #layout>

  return %3 : tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #layout>
}

// Test local-to-local with different grid shapes but same layout properties
#layout_same = #ttcore.metal_layout<logical_shape = 256x256, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>

// CHECK-LABEL: func.func @test_simple_reblock_local
func.func @test_simple_reblock_local(%arg0: tensor<2x2x4x4x!ttcore.tile<32x32, f32>, #layout_same>) -> tensor<4x4x2x2x!ttcore.tile<32x32, f32>, #layout_same> {
  %0 = d2m.empty() : tensor<4x4x2x2x!ttcore.tile<32x32, f32>, #layout_same>

  // CHECK: %[[VIEW:.*]] = d2m.view_layout
  // CHECK: d2m.generic
  // CHECK-SAME: threads = [#d2m.thread<unified>]
  // CHECK: d2m.block_index(0)
  // CHECK: d2m.block_index(1)
  // CHECK: d2m.remote_load %{{.*}} %[[VIEW]][%{{.*}}, %{{.*}}]
  // CHECK-NOT: d2m.dma

  %1 = d2m.to_layout %arg0, %0 : tensor<2x2x4x4x!ttcore.tile<32x32, f32>, #layout_same> into tensor<4x4x2x2x!ttcore.tile<32x32, f32>, #layout_same>
    -> tensor<4x4x2x2x!ttcore.tile<32x32, f32>, #layout_same>

  return %1 : tensor<4x4x2x2x!ttcore.tile<32x32, f32>, #layout_same>
}

// Test minimal local-to-local case: 1x1 to 2x2 grid reblock
#layout_minimal = #ttcore.metal_layout<logical_shape = 128x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>

// CHECK-LABEL: func.func @test_minimal_local_reblock
func.func @test_minimal_local_reblock(%arg0: tensor<1x1x4x4x!ttcore.tile<32x32, f32>, #layout_minimal>) -> tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout_minimal> {
  %0 = d2m.empty() : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout_minimal>

  // Verify that local-to-local transfer uses remote_load (not DMA)
  // CHECK: %[[VIEW:.*]] = d2m.view_layout
  // CHECK: %[[RESULT:.*]] = d2m.generic
  // CHECK-SAME: grid = #ttcore.grid<2x2
  // CHECK-SAME: threads = [#d2m.thread<unified>]
  // CHECK: ^{{.*}}(%[[CB_IN:.*]]: !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>>>, %[[CB_OUT:.*]]: !d2m.cb<tensor<2x2x!ttcore.tile<32x32, f32>>>):
  // CHECK: d2m.block_index(0)
  // CHECK: d2m.block_index(1)
  // CHECK: d2m.remote_load %{{.*}} %[[VIEW]][%{{.*}}, %{{.*}}]
  // CHECK: d2m.yield
  // Verify no DMA operations are generated
  // CHECK-NOT: d2m.dma

  %1 = d2m.to_layout %arg0, %0 : tensor<1x1x4x4x!ttcore.tile<32x32, f32>, #layout_minimal> into tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout_minimal>
    -> tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout_minimal>

  return %1 : tensor<2x2x2x2x!ttcore.tile<32x32, f32>, #layout_minimal>
}
