// RUN: ttmlir-opt --ttcore-register-device --d2m-lower-to-layout --d2m-materialize-view-returns -o %t %s
// RUN: FileCheck %s --input-file=%t

#layout = #ttcore.metal_layout<logical_shape = 1024x1024, dim_alignments = 256x256, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>
#layout2 = #ttcore.metal_layout<logical_shape = 256x768, dim_alignments = 32x256, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>

// Test that verifies the NEW behavior: distribute to 8x8 grid first, then tilize
func.func @tilize(%arg0: tensor<1024x1024xf32>) -> tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout> {
  %0 = d2m.empty() : tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout>

  // CHECK-LABEL: @tilize
  // Verify the operation creates intermediate 8x8 distributed tensor
  // CHECK: %[[TILED:.*]] = d2m.empty() : tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout>
  // CHECK: %[[INTERMEDIATE:.*]] = d2m.empty() : tensor<8x8x128x128xf32, #layout{{[0-9]*}}>
  // Host-to-device transfer now uses dedicated d2m.to_device op
  // CHECK: %[[TO_DEVICE:.*]] = d2m.to_device %arg0, %[[INTERMEDIATE]] layout = #layout{{[0-9]*}} : tensor<1024x1024xf32> into tensor<8x8x128x128xf32, #layout{{[0-9]*}}> -> tensor<8x8x128x128xf32, #layout{{[0-9]*}}>
  // CHECK: %[[RESULT:.*]] = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<8x8>
  // CHECK-SAME: threads = [#d2m.thread<unified>]
  // CHECK-NEXT: ins(%[[TO_DEVICE]] : tensor<8x8x128x128xf32, #layout{{[0-9]*}}>)
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
  // Grid reblocking emits view_layout + generic with load+store pair
  // CHECK: %[[VIEW:.*]] = d2m.view_layout
  // CHECK: d2m.generic
  // CHECK-SAME: threads = [#d2m.thread<unified>]
  // CHECK: d2m.remote_load
  // CHECK: d2m.remote_store

  %1 = d2m.to_layout %arg0, %0 : tensor<1x1x8x24x!ttcore.tile<32x32, f32>, #layout2> into tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout2>
    -> tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout2>

  return %1 : tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout2>
}

// Test untilize with 8x8 distribution
func.func @untilize(%arg0: tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout>) -> tensor<1024x1024xf32> {
  %0 = d2m.empty() : tensor<1024x1024xf32>

  // CHECK-LABEL: @untilize
  // CHECK: %[[HOST:.*]] = d2m.empty() : tensor<1024x1024xf32>
  // CHECK: %[[INTERMEDIATE:.*]] = d2m.empty() : tensor<8x8x128x128xf32, #layout{{[0-9]*}}>
  // CHECK: %[[UNTILIZED:.*]] = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<8x8>
  // CHECK-SAME: threads = [#d2m.thread<unified>]
  // CHECK: d2m.tile_untilize_block
  // Device-to-host transfer now uses dedicated d2m.to_host op
  // CHECK: d2m.to_host %[[UNTILIZED]], %[[HOST]] layout = #layout{{[0-9]*}} : tensor<8x8x128x128xf32, #layout{{[0-9]*}}> into tensor<1024x1024xf32>

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
  // Host-to-device transfer uses d2m.to_device
  // CHECK: d2m.to_device %arg0, %{{.*}} layout = #layout{{[0-9]*}} : tensor<256x768xf32> into tensor<8x8x32x96xf32, #layout{{[0-9]*}}>
  // CHECK: d2m.generic {{{.*}}grid = #ttcore.grid<8x8>{{.*}}threads = [#d2m.thread<unified>]
  // CHECK: d2m.tile_tilize_block
  // CHECK: d2m.generic {{{.*}}grid = #ttcore.grid<8x8>{{.*}}threads = [#d2m.thread<unified>]
  // CHECK: d2m.tile_untilize_block
  // Device-to-host transfer uses d2m.to_host
  // CHECK: d2m.to_host %{{.*}} layout = #layout{{[0-9]*}} : tensor<8x8x32x96xf32, #layout{{[0-9]*}}> into tensor<256x768xf32>

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
  // CHECK: d2m.empty() : tensor<8x8x128x128xf32, #layout{{[0-9]*}}>
  // Host-to-device transfer uses d2m.to_device
  // CHECK: d2m.to_device %arg0, %{{.*}} layout = #layout{{[0-9]*}} : tensor<1024x1024xf32> into tensor<8x8x128x128xf32, #layout{{[0-9]*}}>
  // CHECK: d2m.generic {{{.*}}grid = #ttcore.grid<8x8>{{.*}}threads = [#d2m.thread<unified>]

  %1 = d2m.to_layout %arg0, %0 : tensor<1024x1024xf32> into tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout>
    -> tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout>

  return %1 : tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout>
}

// Test padding changes - different dim_alignments causing different device tensor shapes
#layout_pad32 = #ttcore.metal_layout<logical_shape = 96x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>
#layout_pad64 = #ttcore.metal_layout<logical_shape = 96x128, dim_alignments = 64x64, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>

func.func @padding_change(%arg0: tensor<2x4x32x32xf32, #layout_pad32>) -> tensor<2x2x64x64xf32, #layout_pad64> {
  %0 = d2m.empty() : tensor<2x2x64x64xf32, #layout_pad64>

  // CHECK-LABEL: @padding_change
  // Padding changes (same logical shape, different alignments) emit view_layout + generic with load+store pair
  // CHECK: %[[VIEW:.*]] = d2m.view_layout
  // CHECK: d2m.generic
  // CHECK-SAME: threads = [#d2m.thread<unified>]
  // CHECK: d2m.remote_load
  // CHECK: d2m.remote_store

  %1 = d2m.to_layout %arg0, %0 : tensor<2x4x32x32xf32, #layout_pad32> into tensor<2x2x64x64xf32, #layout_pad64>
    -> tensor<2x2x64x64xf32, #layout_pad64>

  return %1 : tensor<2x2x64x64xf32, #layout_pad64>
}

// Test compound transformations - grid reblock + padding change simultaneously
#layout_src_compound = #ttcore.metal_layout<logical_shape = 64x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>
#layout_dst_compound = #ttcore.metal_layout<logical_shape = 64x128, dim_alignments = 64x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>

func.func @compound_reblock_pad(%arg0: tensor<4x2x32x32xf32, #layout_src_compound>) -> tensor<2x4x64x32xf32, #layout_dst_compound> {
  %0 = d2m.empty() : tensor<2x4x64x32xf32, #layout_dst_compound>

  // CHECK-LABEL: @compound_reblock_pad
  // Combined grid reblock + padding emits view_layout + generic with load+store pair
  // CHECK: %[[VIEW:.*]] = d2m.view_layout
  // CHECK: d2m.generic
  // CHECK-SAME: threads = [#d2m.thread<unified>]
  // CHECK: d2m.remote_load
  // CHECK: d2m.remote_store

  %1 = d2m.to_layout %arg0, %0 : tensor<4x2x32x32xf32, #layout_src_compound> into tensor<2x4x64x32xf32, #layout_dst_compound>
    -> tensor<2x4x64x32xf32, #layout_dst_compound>

  return %1 : tensor<2x4x64x32xf32, #layout_dst_compound>
}

// Test masking with non-undef OOBVal and padding
// logical_shape = 50x50 doesn't align to dim_alignments = 32x32, so padding exists
// OOBVal = zero (not undef) should trigger masking
#layout_mask = #ttcore.metal_layout<logical_shape = 50x50, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, zero, l1, sharded>

func.func @tilize_with_masking(%arg0: tensor<50x50xf32>) -> tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_mask> {
  %0 = d2m.empty() : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_mask>

  // CHECK-LABEL: @tilize_with_masking
  // After tilization, masking should be applied due to non-undef OOBVal + padding
  // CHECK: d2m.tile_tilize_block
  // CHECK: d2m.block_mask
  // CHECK-SAME: <zero>

  %1 = d2m.to_layout %arg0, %0 : tensor<50x50xf32> into tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_mask>
    -> tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_mask>

  return %1 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_mask>
}

// Test masking with non-undef OOBVal and >2D grid (no collapse).
// This exercises the case where collapsed_intervals is empty, producing >2
// physical dimensions.  The mask scratch tensors must have the same grid rank
// as the main operands; a prior bug only appended [1,1] shard dims (always
// producing a 2D-grid mask) which crashed the verifier when gridRank > 2.
#layout_mask_4d = #ttcore.metal_layout<logical_shape = 2x2x50x50, dim_alignments = 1x1x32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, zero, l1, sharded, index_map = map(8)>

func.func @masking_no_collapse_4d(%arg0: tensor<2x2x50x50xf32>) -> tensor<1x1x1x1x2x2x2x2x!ttcore.tile<32x32, f32>, #layout_mask_4d> {
  %0 = d2m.empty() : tensor<1x1x1x1x2x2x2x2x!ttcore.tile<32x32, f32>, #layout_mask_4d>

  // CHECK-LABEL: @masking_no_collapse_4d
  // Tilize then mask with zero OOBVal on a >2D grid (no collapse)
  // CHECK: d2m.tile_tilize_block
  // CHECK: d2m.block_mask
  // CHECK-SAME: <zero>

  %1 = d2m.to_layout %arg0, %0 : tensor<2x2x50x50xf32> into tensor<1x1x1x1x2x2x2x2x!ttcore.tile<32x32, f32>, #layout_mask_4d>
    -> tensor<1x1x1x1x2x2x2x2x!ttcore.tile<32x32, f32>, #layout_mask_4d>

  return %1 : tensor<1x1x1x1x2x2x2x2x!ttcore.tile<32x32, f32>, #layout_mask_4d>
}

// Test chained views with pre-existing index_map
#layout_base_view = #ttcore.metal_layout<logical_shape = 64x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>
#layout_with_view = #ttcore.metal_layout<logical_shape = 64x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>

func.func @chained_view(%arg0: tensor<2x4x32x32xf32, #layout_base_view>) -> tensor<4x2x32x32xf32, #layout_with_view> {
  %0 = d2m.empty() : tensor<4x2x32x32xf32, #layout_with_view>

  // CHECK-LABEL: @chained_view
  // View chaining (no grid change, just index_map addition) emits view_layout + generic with load+store pair
  // CHECK: %[[VIEW:.*]] = d2m.view_layout
  // CHECK: d2m.generic
  // CHECK-SAME: threads = [#d2m.thread<unified>]
  // CHECK: d2m.remote_load
  // CHECK: d2m.remote_store

  %1 = d2m.to_layout %arg0, %0 : tensor<2x4x32x32xf32, #layout_base_view> into tensor<4x2x32x32xf32, #layout_with_view>
    -> tensor<4x2x32x32xf32, #layout_with_view>

  return %1 : tensor<4x2x32x32xf32, #layout_with_view>
}

// Always use interleaved DRAM bounce for tensors on virtual grids

#layout_virtual_dram_aligned = #ttcore.metal_layout<logical_shape = 4x32x32, dim_alignments = 1x32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, undef, l1, sharded>

func.func @test_virtual_dram_bounce_aligned(%arg0: tensor<4x32x32xf32>) -> tensor<4x1x1x1x32x32xf32, #layout_virtual_dram_aligned> {
  // CHECK-LABEL: @test_virtual_dram_bounce_aligned
  // CHECK: %{{.*}} = d2m.empty() : tensor<1x1x128x32xf32, #layout[[DRAM_LAYOUT_ALIGNED:[0-9]*]]
  // CHECK: d2m.to_device {{.*}} layout = #layout[[DRAM_LAYOUT_ALIGNED]]
  %0 = d2m.empty() : tensor<4x1x1x1x32x32xf32, #layout_virtual_dram_aligned>
  %1 = d2m.to_layout %arg0, %0 : tensor<4x32x32xf32> into tensor<4x1x1x1x32x32xf32, #layout_virtual_dram_aligned> -> tensor<4x1x1x1x32x32xf32, #layout_virtual_dram_aligned>
  return %1 : tensor<4x1x1x1x32x32xf32, #layout_virtual_dram_aligned>
}

#layout_virtual_dram_unaligned = #ttcore.metal_layout<logical_shape = 4x128x32, dim_alignments = 1x32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, undef, l1, sharded>

func.func @test_virtual_dram_bounce_unaligned(%arg0: tensor<4x128x32xf32>) -> tensor<4x4x1x1x32x32xf32, #layout_virtual_dram_unaligned> {
  // CHECK-LABEL: @test_virtual_dram_bounce_unaligned
  // CHECK: %{{.*}} = d2m.empty() : tensor<1x1x512x32xf32, #layout[[DRAM_LAYOUT_UNALIGNED:[0-9]*]]
  // CHECK: d2m.to_device {{.*}} layout = #layout[[DRAM_LAYOUT_UNALIGNED]]
  %0 = d2m.empty() : tensor<4x4x1x1x32x32xf32, #layout_virtual_dram_unaligned>
  %1 = d2m.to_layout %arg0, %0 : tensor<4x128x32xf32> into tensor<4x4x1x1x32x32xf32, #layout_virtual_dram_unaligned> -> tensor<4x4x1x1x32x32xf32, #layout_virtual_dram_unaligned>
  return %1 : tensor<4x4x1x1x32x32xf32, #layout_virtual_dram_unaligned>
}
