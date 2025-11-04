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
  // Reblocking is a view since it's expressible via affine maps
  // The view should have a transformation map attached (verified via index_map in result layout)
  // CHECK: d2m.view_layout
  // CHECK: ttir.abs
  // CHECK-NOT: d2m.stream_layout

  %1 = d2m.to_layout %arg0, %0 : tensor<1x1x8x24x!ttcore.tile<32x32, f32>, #layout2> into tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout2>
    -> tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout2>

  // Consume the view with an operation before returning
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
  // CHECK: ttir.relu
  // CHECK-NOT: d2m.stream_layout

  %1 = d2m.to_layout %arg0, %0 : tensor<2x4x32x32xf32, #layout_identity> into tensor<2x4x32x32xf32, #layout_with_transpose> -> tensor<2x4x32x32xf32, #layout_with_transpose>

  // Consume the view before returning
  %2 = d2m.empty() : tensor<2x4x32x32xf32, #layout_with_transpose>
  %3 = "ttir.relu"(%1, %2) : (tensor<2x4x32x32xf32, #layout_with_transpose>, tensor<2x4x32x32xf32, #layout_with_transpose>) -> tensor<2x4x32x32xf32, #layout_with_transpose>

  return %3 : tensor<2x4x32x32xf32, #layout_with_transpose>
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

// Test: Different grid shapes -> different tensor shape -> ViewLayoutOp
// Note: collapsed_intervals are same, but grid is different
func.func @view_grid_change(%arg0: tensor<2x4x32x32xf32, #layout_2d_grid2x4>)
    -> tensor<8x1x256x128xf32, #layout_2d_grid8x1> {
  %0 = d2m.empty() : tensor<8x1x256x128xf32, #layout_2d_grid8x1>

  // CHECK-LABEL: @view_grid_change
  // Tensor shapes differ due to different grid: [2,4,32,32] vs [8,1,256,128]
  // -> should be a view with an affine transformation map
  // CHECK: d2m.view_layout
  // CHECK: ttir.exp
  // CHECK-NOT: d2m.stream_layout

  %1 = d2m.to_layout %arg0, %0 : tensor<2x4x32x32xf32, #layout_2d_grid2x4> into tensor<8x1x256x128xf32, #layout_2d_grid8x1> -> tensor<8x1x256x128xf32, #layout_2d_grid8x1>

  // Consume the view before returning
  %2 = d2m.empty() : tensor<8x1x256x128xf32, #layout_2d_grid8x1>
  %3 = "ttir.exp"(%1, %2) : (tensor<8x1x256x128xf32, #layout_2d_grid8x1>, tensor<8x1x256x128xf32, #layout_2d_grid8x1>) -> tensor<8x1x256x128xf32, #layout_2d_grid8x1>

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

// Test: Different alignments causing different tensor shapes -> ViewLayoutOp
func.func @view_alignment_change(%arg0: tensor<2x4x32x32xf32, #layout_align_32>)
    -> tensor<1x2x64x64xf32, #layout_align_64> {
  %0 = d2m.empty() : tensor<1x2x64x64xf32, #layout_align_64>

  // CHECK-LABEL: @view_alignment_change
  // Tensor shapes differ due to dim_alignments: [2,4,32,32] vs [1,2,64,64]
  // -> should be a view with transformation map
  // CHECK: d2m.view_layout
  // CHECK: ttir.neg
  // CHECK-NOT: d2m.stream_layout

  %1 = d2m.to_layout %arg0, %0 : tensor<2x4x32x32xf32, #layout_align_32> into tensor<1x2x64x64xf32, #layout_align_64> -> tensor<1x2x64x64xf32, #layout_align_64>

  // Consume the view before returning
  %2 = d2m.empty() : tensor<1x2x64x64xf32, #layout_align_64>
  %3 = "ttir.neg"(%1, %2) : (tensor<1x2x64x64xf32, #layout_align_64>, tensor<1x2x64x64xf32, #layout_align_64>) -> tensor<1x2x64x64xf32, #layout_align_64>

  return %3 : tensor<1x2x64x64xf32, #layout_align_64>
}

// -----

// Note: TensorMemoryLayout changes (Interleaved <-> Sharded) are NOT supported
// via ToLayoutOp as they would require actual data movement/reshuffling.
// All layouts in this file use 'sharded' exclusively.

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
  // CHECK: ttir.abs
  // CHECK-NOT: d2m.stream_layout

  %1 = d2m.to_layout %arg0, %0 : tensor<2x4x1x1x!ttcore.tile<32x32, f32>, #layout_tiled_a> into tensor<2x4x1x1x!ttcore.tile<32x32, f32>, #layout_tiled_b> -> tensor<2x4x1x1x!ttcore.tile<32x32, f32>, #layout_tiled_b>

  // Consume the view before returning
  %2 = d2m.empty() : tensor<2x4x1x1x!ttcore.tile<32x32, f32>, #layout_tiled_b>
  %3 = "ttir.abs"(%1, %2) : (tensor<2x4x1x1x!ttcore.tile<32x32, f32>, #layout_tiled_b>, tensor<2x4x1x1x!ttcore.tile<32x32, f32>, #layout_tiled_b>) -> tensor<2x4x1x1x!ttcore.tile<32x32, f32>, #layout_tiled_b>

  return %3 : tensor<2x4x1x1x!ttcore.tile<32x32, f32>, #layout_tiled_b>
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
  // CHECK: ttir.relu
  // CHECK-NOT: d2m.stream_layout

  %1 = d2m.to_layout %arg0, %0 : tensor<2x4x32x32xf32, #layout_base> into tensor<2x4x32x32xf32, #layout_offset_view> -> tensor<2x4x32x32xf32, #layout_offset_view>

  // Consume the view before returning
  %2 = d2m.empty() : tensor<2x4x32x32xf32, #layout_offset_view>
  %3 = "ttir.relu"(%1, %2) : (tensor<2x4x32x32xf32, #layout_offset_view>, tensor<2x4x32x32xf32, #layout_offset_view>) -> tensor<2x4x32x32xf32, #layout_offset_view>

  return %3 : tensor<2x4x32x32xf32, #layout_offset_view>
}

// -----

// Test padding changes: smaller to bigger padding
#layout_small_pad = #ttcore.metal_layout<
  logical_shape = 100x200,
  dim_alignments = 32x32,
  collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>,
  undef,
  l1,
  sharded
>

#layout_big_pad = #ttcore.metal_layout<
  logical_shape = 100x200,
  dim_alignments = 128x128,
  collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>,
  undef,
  l1,
  sharded
>

func.func @view_padding_increase(%arg0: tensor<2x4x32x32xf32, #layout_small_pad>)
    -> tensor<1x2x128x128xf32, #layout_big_pad> {
  %0 = d2m.empty() : tensor<1x2x128x128xf32, #layout_big_pad>

  // CHECK-LABEL: @view_padding_increase
  // Logical shape stays 100x200, but padding increases (32x32 -> 128x128)
  // Device tensor shape changes: [2,4,32,32] -> [1,2,128,128]
  // CHECK: d2m.view_layout
  // CHECK: ttir.sigmoid
  // CHECK-NOT: d2m.stream_layout

  %1 = d2m.to_layout %arg0, %0 : tensor<2x4x32x32xf32, #layout_small_pad> into tensor<1x2x128x128xf32, #layout_big_pad> -> tensor<1x2x128x128xf32, #layout_big_pad>

  %2 = d2m.empty() : tensor<1x2x128x128xf32, #layout_big_pad>
  %3 = "ttir.sigmoid"(%1, %2) : (tensor<1x2x128x128xf32, #layout_big_pad>, tensor<1x2x128x128xf32, #layout_big_pad>) -> tensor<1x2x128x128xf32, #layout_big_pad>

  return %3 : tensor<1x2x128x128xf32, #layout_big_pad>
}

// -----

// Test padding changes: bigger to smaller padding
#layout_large_pad = #ttcore.metal_layout<
  logical_shape = 95x205,
  dim_alignments = 128x128,
  collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>,
  undef,
  l1,
  sharded
>

#layout_medium_pad = #ttcore.metal_layout<
  logical_shape = 95x205,
  dim_alignments = 64x64,
  collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>,
  undef,
  l1,
  sharded
>

func.func @view_padding_decrease(%arg0: tensor<1x2x128x128xf32, #layout_large_pad>)
    -> tensor<2x4x64x64xf32, #layout_medium_pad> {
  %0 = d2m.empty() : tensor<2x4x64x64xf32, #layout_medium_pad>

  // CHECK-LABEL: @view_padding_decrease
  // Logical shape stays 95x205, but padding decreases (128x128 -> 64x64)
  // Device tensor shape changes: [1,2,128,128] -> [2,4,64,64]
  // CHECK: d2m.view_layout
  // CHECK: ttir.tanh
  // CHECK-NOT: d2m.stream_layout

  %1 = d2m.to_layout %arg0, %0 : tensor<1x2x128x128xf32, #layout_large_pad> into tensor<2x4x64x64xf32, #layout_medium_pad> -> tensor<2x4x64x64xf32, #layout_medium_pad>

  %2 = d2m.empty() : tensor<2x4x64x64xf32, #layout_medium_pad>
  %3 = "ttir.tanh"(%1, %2) : (tensor<2x4x64x64xf32, #layout_medium_pad>, tensor<2x4x64x64xf32, #layout_medium_pad>) -> tensor<2x4x64x64xf32, #layout_medium_pad>

  return %3 : tensor<2x4x64x64xf32, #layout_medium_pad>
}

// -----

// Test collapsed intervals: 3D to 2D (collapse first two logical dims)
#layout_3d_logical = #ttcore.metal_layout<
  logical_shape = 64x64x64,
  dim_alignments = 32x32x32,
  collapsed_intervals = dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>,
  undef,
  l1,
  sharded
>

#layout_2d_collapse_first = #ttcore.metal_layout<
  logical_shape = 64x64x64,
  dim_alignments = 32x32x32,
  collapsed_intervals = dense<[[0, 2], [2, 3]]> : tensor<2x2xi64>,
  undef,
  l1,
  sharded
>

func.func @view_collapse_3d_to_2d_first(%arg0: tensor<2x2x2x32x32x32xf32, #layout_3d_logical>)
    -> tensor<4x2x32x32xf32, #layout_2d_collapse_first> {
  %0 = d2m.empty() : tensor<4x2x32x32xf32, #layout_2d_collapse_first>

  // CHECK-LABEL: @view_collapse_3d_to_2d_first
  // Logical shape stays 64x64x64
  // Collapsed intervals: [[0,1],[1,2],[2,3]] (3D) -> [[0,2],[2,3]] (2D, first two dims collapsed)
  // Grid changes from 3D [2,2,2] to 2D [4,2]
  // CHECK: d2m.view_layout
  // CHECK: ttir.log
  // CHECK-NOT: d2m.stream_layout

  %1 = d2m.to_layout %arg0, %0 : tensor<2x2x2x32x32x32xf32, #layout_3d_logical> into tensor<4x2x32x32xf32, #layout_2d_collapse_first> -> tensor<4x2x32x32xf32, #layout_2d_collapse_first>

  %2 = d2m.empty() : tensor<4x2x32x32xf32, #layout_2d_collapse_first>
  %3 = "ttir.log"(%1, %2) : (tensor<4x2x32x32xf32, #layout_2d_collapse_first>, tensor<4x2x32x32xf32, #layout_2d_collapse_first>) -> tensor<4x2x32x32xf32, #layout_2d_collapse_first>

  return %3 : tensor<4x2x32x32xf32, #layout_2d_collapse_first>
}

// -----

// Test collapsed intervals: 3D to 2D (collapse last two logical dims)
#layout_3d = #ttcore.metal_layout<
  logical_shape = 64x64x64,
  dim_alignments = 32x32x32,
  collapsed_intervals = dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>,
  undef,
  l1,
  sharded
>

#layout_2d_collapse_last = #ttcore.metal_layout<
  logical_shape = 64x64x64,
  dim_alignments = 32x32x32,
  collapsed_intervals = dense<[[0, 1], [1, 3]]> : tensor<2x2xi64>,
  undef,
  l1,
  sharded
>

func.func @view_collapse_3d_to_2d_last(%arg0: tensor<2x2x2x32x32x32xf32, #layout_3d>)
    -> tensor<2x4x32x32xf32, #layout_2d_collapse_last> {
  %0 = d2m.empty() : tensor<2x4x32x32xf32, #layout_2d_collapse_last>

  // CHECK-LABEL: @view_collapse_3d_to_2d_last
  // Logical shape stays 64x64x64
  // Collapsed intervals: [[0,1],[1,2],[2,3]] (3D) -> [[0,1],[1,3]] (2D, last two dims collapsed)
  // Grid changes from 3D [2,2,2] to 2D [2,4]
  // CHECK: d2m.view_layout
  // CHECK: ttir.reciprocal
  // CHECK-NOT: d2m.stream_layout

  %1 = d2m.to_layout %arg0, %0 : tensor<2x2x2x32x32x32xf32, #layout_3d> into tensor<2x4x32x32xf32, #layout_2d_collapse_last> -> tensor<2x4x32x32xf32, #layout_2d_collapse_last>

  %2 = d2m.empty() : tensor<2x4x32x32xf32, #layout_2d_collapse_last>
  %3 = "ttir.reciprocal"(%1, %2) : (tensor<2x4x32x32xf32, #layout_2d_collapse_last>, tensor<2x4x32x32xf32, #layout_2d_collapse_last>) -> tensor<2x4x32x32xf32, #layout_2d_collapse_last>

  return %3 : tensor<2x4x32x32xf32, #layout_2d_collapse_last>
}

// -----

// Test collapsed intervals: 2D to 3D (uncollapse)
#layout_2d_src = #ttcore.metal_layout<
  logical_shape = 64x64x64,
  dim_alignments = 32x32x32,
  collapsed_intervals = dense<[[0, 2], [2, 3]]> : tensor<2x2xi64>,
  undef,
  l1,
  sharded
>

#layout_3d_dst = #ttcore.metal_layout<
  logical_shape = 64x64x64,
  dim_alignments = 32x32x32,
  collapsed_intervals = dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>,
  undef,
  l1,
  sharded
>

func.func @view_uncollapse_2d_to_3d(%arg0: tensor<4x2x32x32xf32, #layout_2d_src>)
    -> tensor<2x2x2x32x32x32xf32, #layout_3d_dst> {
  %0 = d2m.empty() : tensor<2x2x2x32x32x32xf32, #layout_3d_dst>

  // CHECK-LABEL: @view_uncollapse_2d_to_3d
  // Logical shape stays 64x64x64
  // Collapsed intervals: [[0,2],[2,3]] (2D) -> [[0,1],[1,2],[2,3]] (3D, uncollapsed)
  // Grid changes from 2D [4,2] to 3D [2,2,2]
  // CHECK: d2m.view_layout
  // CHECK: ttir.rsqrt
  // CHECK-NOT: d2m.stream_layout

  %1 = d2m.to_layout %arg0, %0 : tensor<4x2x32x32xf32, #layout_2d_src> into tensor<2x2x2x32x32x32xf32, #layout_3d_dst> -> tensor<2x2x2x32x32x32xf32, #layout_3d_dst>

  %2 = d2m.empty() : tensor<2x2x2x32x32x32xf32, #layout_3d_dst>
  %3 = "ttir.rsqrt"(%1, %2) : (tensor<2x2x2x32x32x32xf32, #layout_3d_dst>, tensor<2x2x2x32x32x32xf32, #layout_3d_dst>) -> tensor<2x2x2x32x32x32xf32, #layout_3d_dst>

  return %3 : tensor<2x2x2x32x32x32xf32, #layout_3d_dst>
}

// -----

// Test combined: padding + collapse + grid changes
#layout_src_combo = #ttcore.metal_layout<
  logical_shape = 96x96x64,
  dim_alignments = 32x32x32,
  collapsed_intervals = dense<[[0, 1], [1, 2], [2, 3]]> : tensor<3x2xi64>,
  undef,
  l1,
  sharded
>

#layout_dst_combo = #ttcore.metal_layout<
  logical_shape = 96x96x64,
  dim_alignments = 64x32x32,
  collapsed_intervals = dense<[[0, 2], [2, 3]]> : tensor<2x2xi64>,
  undef,
  l1,
  sharded
>

func.func @view_combined_changes(%arg0: tensor<3x3x2x32x32x32xf32, #layout_src_combo>)
    -> tensor<9x2x32x32xf32, #layout_dst_combo> {
  %0 = d2m.empty() : tensor<9x2x32x32xf32, #layout_dst_combo>

  // CHECK-LABEL: @view_combined_changes
  // Combined changes:
  // - Logical shape stays 96x96x64
  // - Padding changes (32x32x32 -> 64x32x32 alignments)
  // - Grid changes from 3D [3,3,2] to 2D [9,2]
  // - Collapse changes (3D -> 2D logical)
  // CHECK: d2m.view_layout
  // CHECK: ttir.abs
  // CHECK-NOT: d2m.stream_layout

  %1 = d2m.to_layout %arg0, %0 : tensor<3x3x2x32x32x32xf32, #layout_src_combo> into tensor<9x2x32x32xf32, #layout_dst_combo> -> tensor<9x2x32x32xf32, #layout_dst_combo>

  %2 = d2m.empty() : tensor<9x2x32x32xf32, #layout_dst_combo>
  %3 = "ttir.abs"(%1, %2) : (tensor<9x2x32x32xf32, #layout_dst_combo>, tensor<9x2x32x32xf32, #layout_dst_combo>) -> tensor<9x2x32x32xf32, #layout_dst_combo>

  return %3 : tensor<9x2x32x32xf32, #layout_dst_combo>
}

// -----

// Test: Chained ToLayoutOps - verify composition through existing index_map
#layout_chain_a = #ttcore.metal_layout<
  logical_shape = 64x64,
  dim_alignments = 32x32,
  collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>,
  undef,
  l1,
  sharded
>

#layout_chain_b = #ttcore.metal_layout<
  logical_shape = 64x64,
  dim_alignments = 64x32,
  collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>,
  undef,
  l1,
  sharded
>

#layout_chain_c = #ttcore.metal_layout<
  logical_shape = 64x64,
  dim_alignments = 32x64,
  collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>,
  undef,
  l1,
  sharded
>

func.func @view_chained_transformations(%arg0: tensor<2x4x32x32xf32, #layout_chain_a>)
    -> tensor<2x4x32x64xf32, #layout_chain_c> {
  // First ToLayoutOp: A -> B (creates a view with index_map)
  %0 = d2m.empty() : tensor<2x4x64x32xf32, #layout_chain_b>
  %1 = d2m.to_layout %arg0, %0 : tensor<2x4x32x32xf32, #layout_chain_a> into tensor<2x4x64x32xf32, #layout_chain_b> -> tensor<2x4x64x32xf32, #layout_chain_b>

  // Second ToLayoutOp: B -> C (should compose through B's index_map)
  %2 = d2m.empty() : tensor<2x4x32x64xf32, #layout_chain_c>
  %3 = d2m.to_layout %1, %2 : tensor<2x4x64x32xf32, #layout_chain_b> into tensor<2x4x32x64xf32, #layout_chain_c> -> tensor<2x4x32x64xf32, #layout_chain_c>

  // CHECK-LABEL: @view_chained_transformations
  // Both transformations should be composed into a single view: A->C
  // This tests that when B has an index_map, the second transformation
  // properly composes through it via buildDeviceToLogicalMap
  // CHECK: d2m.view_layout
  // CHECK: ttir.sigmoid
  // CHECK-NOT: d2m.to_layout

  %4 = d2m.empty() : tensor<2x4x32x64xf32, #layout_chain_c>
  %5 = "ttir.sigmoid"(%3, %4) : (tensor<2x4x32x64xf32, #layout_chain_c>, tensor<2x4x32x64xf32, #layout_chain_c>) -> tensor<2x4x32x64xf32, #layout_chain_c>

  return %5 : tensor<2x4x32x64xf32, #layout_chain_c>
}

// -----

// Test: Chained ToLayoutOps where intermediate is used - both views must exist
#layout_chain2_a = #ttcore.metal_layout<
  logical_shape = 64x64,
  dim_alignments = 32x32,
  collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>,
  undef,
  l1,
  sharded
>

#layout_chain2_b = #ttcore.metal_layout<
  logical_shape = 64x64,
  dim_alignments = 64x32,
  collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>,
  undef,
  l1,
  sharded
>

#layout_chain2_c = #ttcore.metal_layout<
  logical_shape = 64x64,
  dim_alignments = 32x64,
  collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>,
  undef,
  l1,
  sharded
>

func.func @view_chained_both_used(%arg0: tensor<2x4x32x32xf32, #layout_chain2_a>)
    -> (tensor<2x4x64x32xf32, #layout_chain2_b>, tensor<2x4x32x64xf32, #layout_chain2_c>) {
  // First ToLayoutOp: A -> B (creates a view with index_map)
  %0 = d2m.empty() : tensor<2x4x64x32xf32, #layout_chain2_b>
  %1 = d2m.to_layout %arg0, %0 : tensor<2x4x32x32xf32, #layout_chain2_a> into tensor<2x4x64x32xf32, #layout_chain2_b> -> tensor<2x4x64x32xf32, #layout_chain2_b>

  // Second ToLayoutOp: B -> C (should compose through B's index_map)
  %2 = d2m.empty() : tensor<2x4x32x64xf32, #layout_chain2_c>
  %3 = d2m.to_layout %1, %2 : tensor<2x4x64x32xf32, #layout_chain2_b> into tensor<2x4x32x64xf32, #layout_chain2_c> -> tensor<2x4x32x64xf32, #layout_chain2_c>

  // CHECK-LABEL: @view_chained_both_used
  // Both transformations are optimized: both views go back to %arg0 with composed maps
  // First view: A->B with composed index_map
  // CHECK: [[VIEW1:%.*]] = d2m.view_layout %arg0
  // CHECK-SAME: tensor<2x4x32x32xf32, #{{.*}}> -> tensor<2x4x64x32xf32, #{{.*}}>
  //
  // Second view: A->C with fully composed index_map (composes through B's transformation)
  // This tests that buildDeviceToLogicalMap properly handles B's existing index_map
  // CHECK: [[VIEW2:%.*]] = d2m.view_layout %arg0
  // CHECK-SAME: tensor<2x4x32x32xf32, #{{.*}}> -> tensor<2x4x32x64xf32, #{{.*}}>
  //
  // Consume both views
  // CHECK: ttir.relu
  // CHECK: ttir.exp

  // Use the first view
  %4 = d2m.empty() : tensor<2x4x64x32xf32, #layout_chain2_b>
  %5 = "ttir.relu"(%1, %4) : (tensor<2x4x64x32xf32, #layout_chain2_b>, tensor<2x4x64x32xf32, #layout_chain2_b>) -> tensor<2x4x64x32xf32, #layout_chain2_b>

  // Use the second view
  %6 = d2m.empty() : tensor<2x4x32x64xf32, #layout_chain2_c>
  %7 = "ttir.exp"(%3, %6) : (tensor<2x4x32x64xf32, #layout_chain2_c>, tensor<2x4x32x64xf32, #layout_chain2_c>) -> tensor<2x4x32x64xf32, #layout_chain2_c>

  return %5, %7 : tensor<2x4x64x32xf32, #layout_chain2_b>, tensor<2x4x32x64xf32, #layout_chain2_c>
}
