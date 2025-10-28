// RUN: ttmlir-opt --ttcore-register-device --d2m-lower-to-layout -o %t %s
// RUN: FileCheck %s --input-file=%t

#layout = #ttcore.metal_layout<logical_shape = 1024x1024, dim_alignments = 256x256, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded, index_map = map(0)>
#layout2 = #ttcore.metal_layout<logical_shape = 256x768, dim_alignments = 32x256, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded, index_map = map(0)>

// Test that verifies the NEW behavior: distribute to 8x8 grid first, then tilize
func.func @tilize(%arg0: tensor<1024x1024xf32>) -> tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout> {
  %0 = d2m.empty() : tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout>

  // CHECK-LABEL: @tilize
  // Verify the operation creates intermediate 8x8 distributed tensor
  // CHECK: %[[TILED:.*]] = d2m.empty() : tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout>
  // CHECK: %[[INTERMEDIATE:.*]] = d2m.empty() : tensor<8x8x128x128xf32, #layout>
  // CHECK: %[[TO_DEVICE:.*]] = d2m.to_layout %arg0, %[[INTERMEDIATE]] : tensor<1024x1024xf32> into tensor<8x8x128x128xf32, #layout>
  // CHECK: %[[RESULT:.*]] = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<8x8>
  // CHECK-SAME: threads = [#d2m.thread<compute>]
  // CHECK-NEXT: ins(%[[TO_DEVICE]] : tensor<8x8x128x128xf32, #layout>)
  // CHECK-NEXT: outs(%[[TILED]] : tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout>)
  // CHECK: d2m.tile_tilize_block
  // CHECK-NOT: d2m.view_layout
  // CHECK-NOT: threads = [#d2m.thread<datamovement>]

  %1 = d2m.to_layout %arg0, %0 : tensor<1024x1024xf32> into tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout>
    -> tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout>

  return %1 : tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout>
}

// Test reblock operation - redistributing already tiled data
func.func @reblock(%arg0: tensor<1x1x8x24x!ttcore.tile<32x32, f32>, #layout2>) -> tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout2> {
  %0 = d2m.empty() : tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout2>

  // CHECK-LABEL: @reblock
  // Reblocking creates a stream since tensor shapes differ (1x1x8x24 vs 8x8x1x3)
  // The stream should have a transformation map attached (verified via index_map in result layout)
  // CHECK: %{{.*}} = "d2m.stream_layout"
  // CHECK-SAME: -> tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout{{[0-9]+}}>
  // CHECK: "ttir.abs"

  %1 = d2m.to_layout %arg0, %0 : tensor<1x1x8x24x!ttcore.tile<32x32, f32>, #layout2> into tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout2>
    -> tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout2>

  // Consume the stream with an actual operation
  %2 = d2m.empty() : tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout2>
  %3 = "ttir.abs"(%1, %2) : (tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout2>, tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout2>) -> tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout2>

  return %3 : tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout2>
}

// Test untilize with 8x8 distribution
func.func @untilize(%arg0: tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout>) -> tensor<1024x1024xf32> {
  %0 = d2m.empty() : tensor<1024x1024xf32>

  // CHECK-LABEL: @untilize
  // CHECK: %[[HOST:.*]] = d2m.empty() : tensor<1024x1024xf32>
  // CHECK: %[[INTERMEDIATE:.*]] = d2m.empty() : tensor<8x8x128x128xf32, #layout>
  // CHECK: %[[UNTILIZED:.*]] = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<8x8>
  // CHECK-SAME: threads = [#d2m.thread<compute>]
  // CHECK: d2m.tile_untilize_block
  // CHECK: d2m.to_layout %[[UNTILIZED]], %[[HOST]] : tensor<8x8x128x128xf32, #layout> into tensor<1024x1024xf32>

  %1 = d2m.to_layout %arg0, %0 : tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout> into tensor<1024x1024xf32>
    -> tensor<1024x1024xf32>

  return %1 : tensor<1024x1024xf32>
}

// Test compound operation with 8x8 distribution
func.func @compound(%arg0: tensor<256x768xf32>) -> tensor<256x768xf32> {
  %0 = d2m.empty() : tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout2>
  %1 = d2m.empty() : tensor<256x768xf32>

  // CHECK-LABEL: @compound
  // CHECK: d2m.empty() : tensor<8x8x32x96xf32, #layout{{[0-9]*}}>
  // CHECK: d2m.to_layout %arg0, %{{.*}} : tensor<256x768xf32> into tensor<8x8x32x96xf32, #layout{{[0-9]*}}>
  // CHECK: d2m.generic {{{.*}}grid = #ttcore.grid<8x8>{{.*}}threads = [#d2m.thread<compute>]
  // CHECK: d2m.tile_tilize_block
  // CHECK: d2m.generic {{{.*}}grid = #ttcore.grid<8x8>{{.*}}threads = [#d2m.thread<compute>]
  // CHECK: d2m.tile_untilize_block
  // CHECK: d2m.to_layout %{{.*}} : tensor<8x8x32x96xf32, #layout{{[0-9]*}}> into tensor<256x768xf32>

  %2 = d2m.to_layout %arg0, %0 : tensor<256x768xf32> into tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout2>
    -> tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout2>

  %3 = d2m.to_layout %2, %1 : tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout2> into tensor<256x768xf32>
    -> tensor<256x768xf32>

  return %3 : tensor<256x768xf32>
}

// Test case showing what the OLD behavior would look like (for documentation purposes)
// This is what we're moving AWAY from
func.func @old_behavior_example(%arg0: tensor<1024x1024xf32>) -> tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout> {
  %0 = d2m.empty() : tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout>

  // CHECK-LABEL: @old_behavior_example
  // The OLD behavior that we DON'T want would have:
  // 1. Move to 1x1 core: tensor<1024x1024xf32> -> tensor<1x1x1024x1024xf32>
  // 2. Tilize on 1x1: tensor<1x1x1024x1024xf32> -> tensor<1x1x32x32x!ttcore.tile<32x32, f32>>
  // 3. View as 8x8: tensor<1x1x32x32x!ttcore.tile<32x32, f32>> -> tensor<8x8x4x4x!ttcore.tile<32x32, f32>>
  // 4. Distribute with datamovement threads

  // But the new implementation should produce the same output as distributed_tilize:
  // CHECK: d2m.empty() : tensor<8x8x128x128xf32, #layout>
  // CHECK: d2m.to_layout %arg0, %{{.*}} : tensor<1024x1024xf32> into tensor<8x8x128x128xf32, #layout>
  // CHECK: d2m.generic {{{.*}}grid = #ttcore.grid<8x8>{{.*}}threads = [#d2m.thread<compute>]

  %1 = d2m.to_layout %arg0, %0 : tensor<1024x1024xf32> into tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout>
    -> tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout>

  return %1 : tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout>
}

// -----
// New tests for View vs Stream logic

#layout_identity = #ttcore.metal_layout<
  logical_shape = 64x128,
  dim_alignments = 32x32,
  collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>,
  undef,
  l1,
  sharded
>

#layout_with_transpose = #ttcore.metal_layout<
  logical_shape = 64x128,
  dim_alignments = 32x32,
  collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>,
  undef,
  l1,
  sharded,
  index_map = (d0, d1) -> (d1, d0)
>

// Test: Same tensor shape, same memory layout, different index map -> ViewLayoutOp
func.func @view_index_map_change(%arg0: tensor<2x4x32x32xf32, #layout_identity>)
    -> tensor<2x4x32x32xf32, #layout_with_transpose> {
  %0 = d2m.empty() : tensor<2x4x32x32xf32, #layout_with_transpose>

  // CHECK-LABEL: @view_index_map_change
  // Same tensor shape (2x4x32x32), same memory layout (sharded)
  // Only index map differs -> should be a view
  // CHECK: d2m.view_layout
  // CHECK-SAME: reinterpretLayout = true
  // CHECK-NOT: d2m.stream_layout

  %1 = d2m.to_layout %arg0, %0 : tensor<2x4x32x32xf32, #layout_identity> into tensor<2x4x32x32xf32, #layout_with_transpose> -> tensor<2x4x32x32xf32, #layout_with_transpose>
  return %1 : tensor<2x4x32x32xf32, #layout_with_transpose>
}

// -----

#layout_2d_grid2x4 = #ttcore.metal_layout<
  logical_shape = 64x128,
  dim_alignments = 32x32,
  collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>,
  undef,
  l1,
  sharded
>

#layout_2d_grid8x1 = #ttcore.metal_layout<
  logical_shape = 64x128,
  dim_alignments = 32x32,
  collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>,
  undef,
  l1,
  sharded
>

// Test: Different grid shapes -> different tensor shape -> StreamLayoutOp
// Note: collapsed_intervals are same, but grid is different
func.func @stream_grid_change(%arg0: tensor<2x4x32x32xf32, #layout_2d_grid2x4>)
    -> tensor<8x1x256x128xf32, #layout_2d_grid8x1> {
  %0 = d2m.empty() : tensor<8x1x256x128xf32, #layout_2d_grid8x1>

  // CHECK-LABEL: @stream_grid_change
  // Tensor shapes differ due to different grid: [2,4,32,32] vs [8,1,256,128]
  // -> should be a stream with an affine transformation map
  // CHECK: %{{.*}} = "d2m.stream_layout"
  // CHECK-SAME: -> tensor<8x1x256x128xf32, #layout{{[0-9]+}}>
  // CHECK: "ttir.relu"
  // CHECK-NOT: d2m.view_layout

  %1 = d2m.to_layout %arg0, %0 : tensor<2x4x32x32xf32, #layout_2d_grid2x4> into tensor<8x1x256x128xf32, #layout_2d_grid8x1> -> tensor<8x1x256x128xf32, #layout_2d_grid8x1>

  // Consume with a unary op
  %2 = d2m.empty() : tensor<8x1x256x128xf32, #layout_2d_grid8x1>
  %3 = "ttir.relu"(%1, %2) : (tensor<8x1x256x128xf32, #layout_2d_grid8x1>, tensor<8x1x256x128xf32, #layout_2d_grid8x1>) -> tensor<8x1x256x128xf32, #layout_2d_grid8x1>

  return %3 : tensor<8x1x256x128xf32, #layout_2d_grid8x1>
}

// -----

#layout_align_32 = #ttcore.metal_layout<
  logical_shape = 60x120,
  dim_alignments = 32x32,
  collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>,
  undef,
  l1,
  sharded
>

#layout_align_64 = #ttcore.metal_layout<
  logical_shape = 60x120,
  dim_alignments = 64x64,
  collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>,
  undef,
  l1,
  sharded
>

// Test: Different alignments causing different tensor shapes -> StreamLayoutOp
func.func @stream_alignment_change(%arg0: tensor<2x4x32x32xf32, #layout_align_32>)
    -> tensor<1x2x64x64xf32, #layout_align_64> {
  %0 = d2m.empty() : tensor<1x2x64x64xf32, #layout_align_64>

  // CHECK-LABEL: @stream_alignment_change
  // Tensor shapes differ due to dim_alignments: [2,4,32,32] vs [1,2,64,64]
  // -> should be a stream with transformation map
  // CHECK: %{{.*}} = "d2m.stream_layout"
  // CHECK-SAME: -> tensor<1x2x64x64xf32, #layout{{[0-9]+}}>
  // CHECK: "ttir.exp"
  // CHECK-NOT: d2m.view_layout

  %1 = d2m.to_layout %arg0, %0 : tensor<2x4x32x32xf32, #layout_align_32> into tensor<1x2x64x64xf32, #layout_align_64> -> tensor<1x2x64x64xf32, #layout_align_64>

  // Consume with a unary op
  %2 = d2m.empty() : tensor<1x2x64x64xf32, #layout_align_64>
  %3 = "ttir.exp"(%1, %2) : (tensor<1x2x64x64xf32, #layout_align_64>, tensor<1x2x64x64xf32, #layout_align_64>) -> tensor<1x2x64x64xf32, #layout_align_64>

  return %3 : tensor<1x2x64x64xf32, #layout_align_64>
}

// -----

#layout_sharded = #ttcore.metal_layout<
  logical_shape = 64x128,
  dim_alignments = 32x32,
  collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>,
  undef,
  l1,
  sharded
>

#layout_interleaved = #ttcore.metal_layout<
  logical_shape = 64x128,
  dim_alignments = 32x32,
  collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>,
  undef,
  l1,
  interleaved
>

// Test: Different memory layout types -> StreamLayoutOp
func.func @stream_memory_layout_type(%arg0: tensor<2x4x32x32xf32, #layout_sharded>)
    -> tensor<2x4x32x32xf32, #layout_interleaved> {
  %0 = d2m.empty() : tensor<2x4x32x32xf32, #layout_interleaved>

  // CHECK-LABEL: @stream_memory_layout_type
  // Same tensor shape but different memory layout type (sharded vs interleaved)
  // -> should be a stream (requires physical data reordering)
  // CHECK: %{{.*}} = "d2m.stream_layout"
  // CHECK-SAME: -> tensor<2x4x32x32xf32, #layout{{[0-9]+}}>
  // CHECK: "ttir.neg"
  // CHECK-NOT: d2m.view_layout

  %1 = d2m.to_layout %arg0, %0 : tensor<2x4x32x32xf32, #layout_sharded> into tensor<2x4x32x32xf32, #layout_interleaved> -> tensor<2x4x32x32xf32, #layout_interleaved>

  // Consume with a unary op
  %2 = d2m.empty() : tensor<2x4x32x32xf32, #layout_interleaved>
  %3 = "ttir.neg"(%1, %2) : (tensor<2x4x32x32xf32, #layout_interleaved>, tensor<2x4x32x32xf32, #layout_interleaved>) -> tensor<2x4x32x32xf32, #layout_interleaved>

  return %3 : tensor<2x4x32x32xf32, #layout_interleaved>
}

// -----

#layout_tiled_a = #ttcore.metal_layout<
  logical_shape = 64x128,
  dim_alignments = 32x32,
  collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>,
  undef,
  l1,
  sharded
>

#layout_tiled_b = #ttcore.metal_layout<
  logical_shape = 64x128,
  dim_alignments = 32x32,
  collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>,
  undef,
  l1,
  sharded,
  index_map = (d0, d1) -> (d1, d0)
>

// Test: Tiled tensors with only index map change -> ViewLayoutOp
func.func @view_tiled_transpose(%arg0: tensor<2x4x1x1x!ttcore.tile<32x32, f32>, #layout_tiled_a>)
    -> tensor<2x4x1x1x!ttcore.tile<32x32, f32>, #layout_tiled_b> {
  %0 = d2m.empty() : tensor<2x4x1x1x!ttcore.tile<32x32, f32>, #layout_tiled_b>

  // CHECK-LABEL: @view_tiled_transpose
  // Same tensor shape, same memory layout, only index map differs
  // Works for tiled tensors too -> should be a view
  // CHECK: d2m.view_layout
  // CHECK-SAME: reinterpretLayout = true
  // CHECK-NOT: d2m.stream_layout

  %1 = d2m.to_layout %arg0, %0 : tensor<2x4x1x1x!ttcore.tile<32x32, f32>, #layout_tiled_a> into tensor<2x4x1x1x!ttcore.tile<32x32, f32>, #layout_tiled_b> -> tensor<2x4x1x1x!ttcore.tile<32x32, f32>, #layout_tiled_b>
  return %1 : tensor<2x4x1x1x!ttcore.tile<32x32, f32>, #layout_tiled_b>
}

// -----

#layout_base = #ttcore.metal_layout<
  logical_shape = 64x128,
  dim_alignments = 32x32,
  collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>,
  undef,
  l1,
  sharded
>

#layout_offset_view = #ttcore.metal_layout<
  logical_shape = 64x128,
  dim_alignments = 32x32,
  collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>,
  undef,
  l1,
  sharded,
  index_map = (d0, d1) -> (d0 + 8, d1 + 16)
>

// Test: View with offset index map
func.func @view_with_offset(%arg0: tensor<2x4x32x32xf32, #layout_base>)
    -> tensor<2x4x32x32xf32, #layout_offset_view> {
  %0 = d2m.empty() : tensor<2x4x32x32xf32, #layout_offset_view>

  // CHECK-LABEL: @view_with_offset
  // Same tensor shape, adding an offset view
  // CHECK: d2m.view_layout
  // CHECK-SAME: reinterpretLayout = true
  // CHECK-NOT: d2m.stream_layout

  %1 = d2m.to_layout %arg0, %0 : tensor<2x4x32x32xf32, #layout_base> into tensor<2x4x32x32xf32, #layout_offset_view> -> tensor<2x4x32x32xf32, #layout_offset_view>
  return %1 : tensor<2x4x32x32xf32, #layout_offset_view>
}
