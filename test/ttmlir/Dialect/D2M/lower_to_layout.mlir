// RUN: ttmlir-opt --ttcore-register-device --d2m-lower-to-layout --d2m-materialize-view-returns -o %t %s
// RUN: FileCheck %s --input-file=%t

#layout = #ttcore.metal_layout<logical_shape = 1024x1024, dim_alignments = 256x256, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#layout_dram = #ttcore.metal_layout<logical_shape = 1024x1024, dim_alignments = 256x256, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, dram, sharded>
#layout2 = #ttcore.metal_layout<logical_shape = 256x768, dim_alignments = 32x256, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>

// Distribute to the target grid before tilization.
func.func @tilize(%arg0: tensor<1024x1024xf32>) -> tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout> {
  %0 = d2m.empty() : tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout>

  // CHECK-LABEL: @tilize
  // CHECK: %[[TILED:.*]] = d2m.empty() : tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout>
  // CHECK: %[[INTERMEDIATE:.*]] = d2m.empty() : tensor<8x8x128x128xf32, #layout{{[0-9]*}}>
  // Host-to-device transfer uses d2m.to_device.
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
  // Device-to-host transfer uses d2m.to_host.
  // CHECK: d2m.to_host %[[UNTILIZED]], %[[HOST]] layout = #layout{{[0-9]*}} : tensor<8x8x128x128xf32, #layout{{[0-9]*}}> into tensor<1024x1024xf32>

  %1 = d2m.to_layout %arg0, %0 : tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout> into tensor<1024x1024xf32>
    -> tensor<1024x1024xf32>

  return %1 : tensor<1024x1024xf32>
}

func.func @tilize_dram_to_l1(%arg0: tensor<8x8x128x128xf32, #layout_dram>) -> tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout> {
  // CHECK-LABEL: @tilize_dram_to_l1
  // CHECK: %[[L1_TILED:.*]] = d2m.empty() : tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #[[L1_LAYOUT:.*]]>
  // CHECK: %[[L1_SCALAR:.*]] = d2m.empty() : tensor<8x8x128x128xf32, #[[L1_LAYOUT]]>
  // CHECK: %[[DRAM_TO_L1:.*]] = d2m.generic
  // CHECK-NEXT: ins(%arg0 : tensor<8x8x128x128xf32, #[[DRAM_LAYOUT:.*]]>)
  // CHECK-NEXT: outs(%[[L1_SCALAR]] : tensor<8x8x128x128xf32, #[[L1_LAYOUT]]>)
  // CHECK: d2m.remote_load
  // CHECK: d2m.remote_store %[[L1_SCALAR]]
  // CHECK: %[[TILIZE_L1:.*]] = d2m.generic
  // CHECK-NEXT: ins(%[[DRAM_TO_L1]] : tensor<8x8x128x128xf32, #[[L1_LAYOUT]]>)
  // CHECK-NEXT: outs(%[[L1_TILED]] : tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #[[L1_LAYOUT]]>)
  // CHECK: d2m.remote_load %{{.+}} %[[DRAM_TO_L1]]
  // CHECK: d2m.tile_tilize_block
  // CHECK: d2m.remote_store %[[L1_TILED]]
  %0 = d2m.empty() : tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout>
  %1 = d2m.to_layout %arg0, %0 : tensor<8x8x128x128xf32, #layout_dram> into tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout> -> tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout>
  return %1 : tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout>
}

// Test untilize from DRAM to host is a direct transfer.
func.func @untilize_dram_to_host(%arg0: tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout_dram>) -> tensor<1024x1024xf32> {
  %0 = d2m.empty() : tensor<1024x1024xf32>

  // CHECK-LABEL: @untilize_dram_to_host
  // CHECK: %[[DRAM_SCALAR:.*]] = d2m.empty() : tensor<8x8x128x128xf32, #[[DRAM_LAYOUT:.*]]>
  // CHECK: %[[DRAM_TO_DRAM:.*]] = d2m.generic
  // CHECK-NEXT: ins(%arg0 : tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #[[DRAM_LAYOUT]]>)
  // CHECK-NEXT: outs(%[[DRAM_SCALAR]] : tensor<8x8x128x128xf32, #[[DRAM_LAYOUT]]>)
  // CHECK-NOT: d2m.generic
  // CHECK: d2m.remote_load
  // CHECK: d2m.tile_untilize_block
  // CHECK: d2m.remote_store %[[DRAM_SCALAR]]
  // CHECK: d2m.to_host %[[DRAM_TO_DRAM]]
  %1 = d2m.to_layout %arg0, %0 : tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout_dram> into tensor<1024x1024xf32>
    -> tensor<1024x1024xf32>

  return %1 : tensor<1024x1024xf32>
}

// Test compound operation with 8x8 distribution
func.func @compound(%arg0: tensor<256x768xf32>) -> tensor<256x768xf32> {
  %0 = d2m.empty() : tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout2>
  %1 = d2m.empty() : tensor<256x768xf32>

  // CHECK-LABEL: @compound
  // CHECK: d2m.empty() : tensor<8x8x32x96xf32, #layout{{[0-9]*}}>
  // Host-to-device transfer uses d2m.to_device.
  // CHECK: d2m.to_device %arg0, %{{.*}} layout = #layout{{[0-9]*}} : tensor<256x768xf32> into tensor<8x8x32x96xf32, #layout{{[0-9]*}}>
  // CHECK: d2m.generic {{{.*}}grid = #ttcore.grid<8x8>{{.*}}threads = [#d2m.thread<unified>]
  // CHECK: d2m.tile_tilize_block
  // CHECK: d2m.generic {{{.*}}grid = #ttcore.grid<8x8>{{.*}}threads = [#d2m.thread<unified>]
  // CHECK: d2m.tile_untilize_block
  // Device-to-host transfer uses d2m.to_host.
  // CHECK: d2m.to_host %{{.*}} layout = #layout{{[0-9]*}} : tensor<8x8x32x96xf32, #layout{{[0-9]*}}> into tensor<256x768xf32>

  %2 = d2m.to_layout %arg0, %0 : tensor<256x768xf32> into tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout2>
    -> tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout2>

  %3 = d2m.to_layout %2, %1 : tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout2> into tensor<256x768xf32>
    -> tensor<256x768xf32>

  return %3 : tensor<256x768xf32>
}

// Preserve the distributed tilize lowering path.
func.func @old_behavior_example(%arg0: tensor<1024x1024xf32>) -> tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout> {
  %0 = d2m.empty() : tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout>

  // CHECK-LABEL: @old_behavior_example
  // CHECK: d2m.empty() : tensor<8x8x128x128xf32, #layout{{[0-9]*}}>
  // Host-to-device transfer uses d2m.to_device.
  // CHECK: d2m.to_device %arg0, %{{.*}} layout = #layout{{[0-9]*}} : tensor<1024x1024xf32> into tensor<8x8x128x128xf32, #layout{{[0-9]*}}>
  // CHECK: d2m.generic {{{.*}}grid = #ttcore.grid<8x8>{{.*}}threads = [#d2m.thread<unified>]

  %1 = d2m.to_layout %arg0, %0 : tensor<1024x1024xf32> into tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout>
    -> tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout>

  return %1 : tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout>
}

// Test padding changes - different dim_alignments causing different device tensor shapes
#layout_pad32 = #ttcore.metal_layout<logical_shape = 96x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#layout_pad64 = #ttcore.metal_layout<logical_shape = 96x128, dim_alignments = 64x64, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>

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
#layout_src_compound = #ttcore.metal_layout<logical_shape = 64x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#layout_dst_compound = #ttcore.metal_layout<logical_shape = 64x128, dim_alignments = 64x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>

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
// OOBVal = zero triggers masking when padding is present.
#layout_mask = #ttcore.metal_layout<logical_shape = 50x50, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>

func.func @tilize_with_masking(%arg0: tensor<50x50xf32>) -> tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_mask> {
  %0 = d2m.empty() : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_mask>

  // CHECK-LABEL: @tilize_with_masking
  // Masking runs after tilization.
  // CHECK: d2m.tile_tilize_block
  // CHECK: d2m.mask
  // CHECK-SAME: logical_shape = [50, 50]
  // CHECK-SAME: fill_value = <zero>

  %1 = d2m.to_layout %arg0, %0 : tensor<50x50xf32> into tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_mask>
    -> tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_mask>
  %2 = d2m.empty() : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_mask>
  %3 = d2m.mask %1, %2 logical_shape = [50, 50] fill_value = <zero> : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_mask> into tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_mask> -> tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_mask>

  return %3 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_mask>
}

// Test masking with non-undef OOBVal and >2D grid (no collapse).
// Mask scratch tensors must keep the same grid rank as the main operands.
#layout_mask_4d = #ttcore.metal_layout<logical_shape = 2x2x50x50, dim_alignments = 1x1x32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, l1, sharded>

func.func @masking_no_collapse_4d(%arg0: tensor<2x2x50x50xf32>) -> tensor<1x1x1x1x2x2x2x2x!ttcore.tile<32x32, f32>, #layout_mask_4d> {
  %0 = d2m.empty() : tensor<1x1x1x1x2x2x2x2x!ttcore.tile<32x32, f32>, #layout_mask_4d>

  // CHECK-LABEL: @masking_no_collapse_4d
  // Tilize, then mask with zero OOBVal on a >2D grid.
  // CHECK: d2m.tile_tilize_block
  // CHECK: d2m.mask
  // CHECK-SAME: logical_shape = [2, 2, 50, 50]
  // CHECK-SAME: fill_value = <zero>

  %1 = d2m.to_layout %arg0, %0 : tensor<2x2x50x50xf32> into tensor<1x1x1x1x2x2x2x2x!ttcore.tile<32x32, f32>, #layout_mask_4d>
    -> tensor<1x1x1x1x2x2x2x2x!ttcore.tile<32x32, f32>, #layout_mask_4d>
  %2 = d2m.empty() : tensor<1x1x1x1x2x2x2x2x!ttcore.tile<32x32, f32>, #layout_mask_4d>
  %3 = d2m.mask %1, %2 logical_shape = [2, 2, 50, 50] fill_value = <zero> : tensor<1x1x1x1x2x2x2x2x!ttcore.tile<32x32, f32>, #layout_mask_4d> into tensor<1x1x1x1x2x2x2x2x!ttcore.tile<32x32, f32>, #layout_mask_4d> -> tensor<1x1x1x1x2x2x2x2x!ttcore.tile<32x32, f32>, #layout_mask_4d>

  return %3 : tensor<1x1x1x1x2x2x2x2x!ttcore.tile<32x32, f32>, #layout_mask_4d>
}

// Test chained views with grid reshaping
#layout_base_view = #ttcore.metal_layout<logical_shape = 64x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#layout_with_view = #ttcore.metal_layout<logical_shape = 64x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>

func.func @chained_view(%arg0: tensor<2x4x32x32xf32, #layout_base_view>) -> tensor<4x2x32x32xf32, #layout_with_view> {
  %0 = d2m.empty() : tensor<4x2x32x32xf32, #layout_with_view>

  // CHECK-LABEL: @chained_view
  // View chaining (grid reshaping) emits view_layout + generic with load+store pair
  // CHECK: %[[VIEW:.*]] = d2m.view_layout
  // CHECK: d2m.generic
  // CHECK-SAME: threads = [#d2m.thread<unified>]
  // CHECK: d2m.remote_load
  // CHECK: d2m.remote_store

  %1 = d2m.to_layout %arg0, %0 : tensor<2x4x32x32xf32, #layout_base_view> into tensor<4x2x32x32xf32, #layout_with_view>
    -> tensor<4x2x32x32xf32, #layout_with_view>

  return %1 : tensor<4x2x32x32xf32, #layout_with_view>
}

// Host transfers for virtual grids stay on the native virtual L1 layout.

#layout_virtual_dram_aligned = #ttcore.metal_layout<logical_shape = 4x32x32, dim_alignments = 1x32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, l1, sharded>

func.func @test_virtual_dram_bounce_aligned(%arg0: tensor<4x32x32xf32>) -> tensor<4x1x1x1x32x32xf32, #layout_virtual_dram_aligned> {
  // CHECK-LABEL: @test_virtual_dram_bounce_aligned
  // CHECK-NOT: tensor<1x1x128x32xf32
  // CHECK: %[[DST:.*]] = d2m.empty() : tensor<4x1x1x1x32x32xf32, #layout[[VIRTUAL_LAYOUT_ALIGNED:[0-9]*]]
  // CHECK: d2m.to_device %arg0, %[[DST]] layout = #layout[[VIRTUAL_LAYOUT_ALIGNED]]
  %0 = d2m.empty() : tensor<4x1x1x1x32x32xf32, #layout_virtual_dram_aligned>
  %1 = d2m.to_layout %arg0, %0 : tensor<4x32x32xf32> into tensor<4x1x1x1x32x32xf32, #layout_virtual_dram_aligned> -> tensor<4x1x1x1x32x32xf32, #layout_virtual_dram_aligned>
  return %1 : tensor<4x1x1x1x32x32xf32, #layout_virtual_dram_aligned>
}

#layout_virtual_dram_unaligned = #ttcore.metal_layout<logical_shape = 4x128x32, dim_alignments = 1x32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, l1, sharded>

func.func @test_virtual_dram_bounce_unaligned(%arg0: tensor<4x128x32xf32>) -> tensor<4x4x1x1x32x32xf32, #layout_virtual_dram_unaligned> {
  // CHECK-LABEL: @test_virtual_dram_bounce_unaligned
  // CHECK-NOT: tensor<1x1x512x32xf32
  // CHECK: %[[DST:.*]] = d2m.empty() : tensor<4x4x1x1x32x32xf32, #layout[[VIRTUAL_LAYOUT_UNALIGNED:[0-9]*]]
  // CHECK: d2m.to_device %arg0, %[[DST]] layout = #layout[[VIRTUAL_LAYOUT_UNALIGNED]]
  %0 = d2m.empty() : tensor<4x4x1x1x32x32xf32, #layout_virtual_dram_unaligned>
  %1 = d2m.to_layout %arg0, %0 : tensor<4x128x32xf32> into tensor<4x4x1x1x32x32xf32, #layout_virtual_dram_unaligned> -> tensor<4x4x1x1x32x32xf32, #layout_virtual_dram_unaligned>
  return %1 : tensor<4x4x1x1x32x32xf32, #layout_virtual_dram_unaligned>
}

#layout_indivisible_bounce_grid = #ttcore.metal_layout<logical_shape = 19x160x32, dim_alignments = 1x32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, l1, sharded>

func.func @test_uncollapsed_indivisible_bounce_grid(%arg0: tensor<1x5x1x19x32x32xbf16, #layout_indivisible_bounce_grid>) -> tensor<19x160x32xbf16> {
  // CHECK-LABEL: @test_uncollapsed_indivisible_bounce_grid
  // CHECK-NOT: d2m.view_layout
  // CHECK: d2m.to_host {{.*}} tensor<1x5x1x19x32x32xbf16, #{{.*}}> into tensor<19x160x32xbf16> -> tensor<19x160x32xbf16>
  %0 = d2m.empty() : tensor<19x160x32xbf16>
  %1 = d2m.to_layout %arg0, %0 : tensor<1x5x1x19x32x32xbf16, #layout_indivisible_bounce_grid> into tensor<19x160x32xbf16> -> tensor<19x160x32xbf16>
  return %1 : tensor<19x160x32xbf16>
}

// Keep uncollapsed grids intact across tiled format conversion.
#layout_tm_src = #ttcore.metal_layout<logical_shape = 5x1024x64, dim_alignments = 1x32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, l1, sharded>
#layout_tm_dst = #ttcore.metal_layout<logical_shape = 5x1024x64, dim_alignments = 1x256x32, collapsed_intervals = dense<> : tensor<0x2xi64>, l1, sharded>

func.func @complex_tiled_mapping_preserves_untilize_shape(%arg0: tensor<1x1x1x5x32x2x!ttcore.tile<32x32, f32>, #layout_tm_src>) -> tensor<1x32x2x5x1x1x!ttcore.tile<32x32, f32>, #layout_tm_dst> {
  // CHECK-LABEL: @complex_tiled_mapping_preserves_untilize_shape
  // CHECK: d2m.tile_untilize_block
  // CHECK-SAME: (tensor<5x32x2x!ttcore.tile<32x32, f32>>, tensor<5x1024x64xf32>) -> tensor<5x1024x64xf32>
  %0 = d2m.empty() : tensor<1x32x2x5x1x1x!ttcore.tile<32x32, f32>, #layout_tm_dst>
  %1 = d2m.to_layout %arg0, %0 : tensor<1x1x1x5x32x2x!ttcore.tile<32x32, f32>, #layout_tm_src> into tensor<1x32x2x5x1x1x!ttcore.tile<32x32, f32>, #layout_tm_dst> -> tensor<1x32x2x5x1x1x!ttcore.tile<32x32, f32>, #layout_tm_dst>
  return %1 : tensor<1x32x2x5x1x1x!ttcore.tile<32x32, f32>, #layout_tm_dst>
}

#rank6_vgm_forward = affine_map<(d0, d1, d2, d3, d4, d5) -> (0, 0, d3, d4, d5)>
#rank6_vgm_inverse = affine_map<(d0, d1) -> (0, 0, 0, 0)>
#rank8_view = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d1, d2, d3, (d4 * 524288 + d5 * 4096 + d6 * 128 + d7) floordiv 524288, ((d4 * 524288 + d5 * 4096 + d6 * 128 + d7) mod 524288) floordiv 4096, (d4 * 524288 + d5 * 4096 + d6 * 128 + d7) mod 4096)>
#rank_incompatible_vgm_src = #ttcore.metal_layout<logical_shape = 1x128x4096, dim_alignments = 1x32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, l1, sharded>
#rank_incompatible_vgm_view = #ttcore.metal_layout<logical_shape = 1x128x32x128, dim_alignments = 1x1x32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, l1, sharded>
#rank_incompatible_vgm_dst = #ttcore.metal_layout<logical_shape = 1x128x32x128, dim_alignments = 256x1x32x32, collapsed_intervals = dense<[[0, 3], [3, 4]]> : tensor<2x2xi64>, l1, sharded>

func.func @tilize_ignores_rank_incompatible_input_vgm() -> tensor<8x4x16x1x!ttcore.tile<32x32, f32>, #rank_incompatible_vgm_dst> {
  // CHECK-LABEL: @tilize_ignores_rank_incompatible_input_vgm
  // CHECK: d2m.empty() {virtualGridForwardMapping = #map{{[0-9]+}}, virtualGridInverseMapping = #map{{[0-9]+}}} : tensor<1x1x1x1x128x4096xf32, #layout{{[0-9]+}}>
  // CHECK: d2m.empty() : tensor<8x4x512x32xf32, #layout{{[0-9]+}}>
  // CHECK: d2m.generic
  // CHECK-SAME: grid = #ttcore.grid<8x4>
  // CHECK: d2m.tile_tilize_block
  %0 = d2m.empty() {virtualGridForwardMapping = #rank6_vgm_forward, virtualGridInverseMapping = #rank6_vgm_inverse} : tensor<1x1x1x1x128x4096xf32, #rank_incompatible_vgm_src>
  %view = d2m.view_layout %0 remapping = #rank8_view : tensor<1x1x1x1x128x4096xf32, #rank_incompatible_vgm_src> -> tensor<1x1x1x1x1x128x32x128xf32, #rank_incompatible_vgm_view>
  %1 = d2m.empty() : tensor<8x4x16x1x!ttcore.tile<32x32, f32>, #rank_incompatible_vgm_dst>
  %2 = d2m.to_layout %view, %1 : tensor<1x1x1x1x1x128x32x128xf32, #rank_incompatible_vgm_view> into tensor<8x4x16x1x!ttcore.tile<32x32, f32>, #rank_incompatible_vgm_dst> -> tensor<8x4x16x1x!ttcore.tile<32x32, f32>, #rank_incompatible_vgm_dst>
  return %2 : tensor<8x4x16x1x!ttcore.tile<32x32, f32>, #rank_incompatible_vgm_dst>
}


#layer_tile_with_vgm_dst = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
func.func @tilize_with_vgm(%arg0: tensor<32x32xbf16>) -> tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layer_tile_with_vgm_dst> {
  // CHECK-LABEL: @tilize_with_vgm
  // Propagate output VGM to the host-to-device intermediate.
  // CHECK: %[[MID:.*]] = d2m.empty() {virtualGridForwardMapping = #map{{[0-9]+}}, virtualGridInverseMapping = #map{{[0-9]+}}} : tensor<1x1x32x32xbf16, #layout{{[0-9]+}}>
  // CHECK: %[[TODEV:.*]] = d2m.to_device %arg0, %[[MID]] layout = #layout{{[0-9]+}} : tensor<32x32xbf16> into tensor<1x1x32x32xbf16, #layout{{[0-9]+}}>
  %0 = d2m.empty() {virtualGridForwardMapping = affine_map<(d0, d1, d2, d3) -> (d0 + 1, d1 + 1, d2, d3)>, virtualGridInverseMapping = affine_map<(d0, d1) -> (0, d0 - 1, d1 - 1)>} : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layer_tile_with_vgm_dst>
  %1 = d2m.to_layout %arg0, %0 : tensor<32x32xbf16> into tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layer_tile_with_vgm_dst> -> tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layer_tile_with_vgm_dst>
  return %1 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layer_tile_with_vgm_dst>
}

func.func @tilize_without_vgm(%arg0: tensor<32x32xbf16>) -> tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layer_tile_with_vgm_dst> {
  // CHECK-LABEL: @tilize_without_vgm
  // Without source/output VGM, use the fallback EmptyOp path.
  // CHECK: %[[MID:.*]] = d2m.empty() : tensor<1x1x32x32xbf16, #layout{{[0-9]+}}>
  // CHECK: %[[TODEV:.*]] = d2m.to_device %arg0, %[[MID]] layout = #layout{{[0-9]+}} : tensor<32x32xbf16> into tensor<1x1x32x32xbf16, #layout{{[0-9]+}}>
  // CHECK: d2m.tile_tilize_block
  %0 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layer_tile_with_vgm_dst>
  %1 = d2m.to_layout %arg0, %0 : tensor<32x32xbf16> into tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layer_tile_with_vgm_dst> -> tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layer_tile_with_vgm_dst>
  return %1 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layer_tile_with_vgm_dst>
}

func.func @tilize_with_vgm_custom_map(%arg0: tensor<32x32xbf16>) -> tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layer_tile_with_vgm_dst> {
  // CHECK-LABEL: @tilize_with_vgm_custom_map
  // CHECK: %[[MID:.*]] = d2m.empty() {virtualGridForwardMapping = #map{{[0-9]+}}, virtualGridInverseMapping = #map{{[0-9]+}}} : tensor<1x1x32x32xbf16, #layout{{[0-9]+}}>
  // CHECK: %[[TODEV:.*]] = d2m.to_device %arg0, %[[MID]] layout = #layout{{[0-9]+}} : tensor<32x32xbf16> into tensor<1x1x32x32xbf16, #layout{{[0-9]+}}>
  // CHECK: grid = #ttcore.grid<1x1, virt_to_physical_map = (d0, d1) -> (0, d0 + 2, d1 + 3), physical_to_virt_map = (d0, d1) -> (0, d0 - 2, d1 - 3)>
  %0 = d2m.empty() {virtualGridForwardMapping = affine_map<(d0, d1, d2, d3) -> (d0 + 2, d1 + 3, d2, d3)>, virtualGridInverseMapping = affine_map<(d0, d1) -> (0, d0 - 2, d1 - 3)>} : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layer_tile_with_vgm_dst>
  %1 = d2m.to_layout %arg0, %0 : tensor<32x32xbf16> into tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layer_tile_with_vgm_dst> -> tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layer_tile_with_vgm_dst>
  return %1 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layer_tile_with_vgm_dst>
}

func.func @tilize_output_only_vgm_fallback(%arg0: tensor<32x32xbf16>) -> tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layer_tile_with_vgm_dst> {
  // CHECK-LABEL: @tilize_output_only_vgm_fallback
  // Output VGM seeds the host-to-device intermediate.
  // CHECK: %[[MID:.*]] = d2m.empty() {virtualGridForwardMapping = #map{{[0-9]+}}, virtualGridInverseMapping = #map{{[0-9]+}}} : tensor<1x1x32x32xbf16, #layout{{[0-9]+}}>
  // CHECK: %[[TODEV:.*]] = d2m.to_device %arg0, %[[MID]] layout = #layout{{[0-9]+}} : tensor<32x32xbf16> into tensor<1x1x32x32xbf16, #layout{{[0-9]+}}>
  %0 = d2m.empty() {virtualGridForwardMapping = affine_map<(d0, d1, d2, d3) -> (d0 + 4, d1 + 5, d2, d3)>, virtualGridInverseMapping = affine_map<(d0, d1) -> (0, d0 - 4, d1 - 5)>} : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layer_tile_with_vgm_dst>
  %1 = d2m.to_layout %arg0, %0 : tensor<32x32xbf16> into tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layer_tile_with_vgm_dst> -> tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layer_tile_with_vgm_dst>
  return %1 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layer_tile_with_vgm_dst>
}

func.func @untilize_input_only_vgm_fallback() -> tensor<32x32xbf16> {
  // CHECK-LABEL: @untilize_input_only_vgm_fallback
  // Input VGM seeds the untilize intermediate.
  // CHECK: %[[MID:.*]] = d2m.empty() {virtualGridForwardMapping = #map{{[0-9]+}}, virtualGridInverseMapping = #map{{[0-9]+}}} : tensor<1x1x32x32xbf16, #layout{{[0-9]+}}>
  // CHECK: outs(%[[MID]] : tensor<1x1x32x32xbf16, #layout{{[0-9]+}}>)
  // CHECK: d2m.tile_untilize_block
  // CHECK: d2m.to_host
  %0 = d2m.empty() {virtualGridForwardMapping = affine_map<(d0, d1, d2, d3) -> (d0 + 6, d1 + 7, d2, d3)>, virtualGridInverseMapping = affine_map<(d0, d1) -> (0, d0 - 6, d1 - 7)>} : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layer_tile_with_vgm_dst>
  %1 = d2m.empty() : tensor<32x32xbf16>
  %2 = d2m.to_layout %0, %1 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layer_tile_with_vgm_dst> into tensor<32x32xbf16> -> tensor<32x32xbf16>
  return %2 : tensor<32x32xbf16>
}

#host_to_dram_with_vgm_dst = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, dram, sharded>

func.func @host_to_dram_with_vgm(%arg0: tensor<64x64xbf16>) -> tensor<2x2x32x32xbf16, #host_to_dram_with_vgm_dst> {
  // CHECK-LABEL: @host_to_dram_with_vgm
  // CHECK: %[[DRAM:.*]] = d2m.empty() {virtualGridForwardMapping = #map{{[0-9]+}}, virtualGridInverseMapping = #map{{[0-9]+}}} : tensor<2x2x32x32xbf16, #layout{{[0-9]+}}>
  // CHECK: %[[TODEV:.*]] = d2m.to_device %arg0, %[[DRAM]] layout = #layout{{[0-9]+}} : tensor<64x64xbf16> into tensor<2x2x32x32xbf16, #layout{{[0-9]+}}>
  // CHECK-NOT: d2m.generic
  // CHECK: return %[[TODEV]]
  %0 = d2m.empty() {virtualGridForwardMapping = affine_map<(d0, d1, d2, d3) -> (d0 + 1, d1 + 1, d2, d3)>, virtualGridInverseMapping = affine_map<(d0, d1) -> (0, d0 - 1, d1 - 1)>} : tensor<2x2x32x32xbf16, #host_to_dram_with_vgm_dst>
  %1 = d2m.to_layout %arg0, %0 : tensor<64x64xbf16> into tensor<2x2x32x32xbf16, #host_to_dram_with_vgm_dst> -> tensor<2x2x32x32xbf16, #host_to_dram_with_vgm_dst>
  return %1 : tensor<2x2x32x32xbf16, #host_to_dram_with_vgm_dst>
}

func.func @dram_to_dram_vgm_change(%arg0: tensor<2x2x32x32xbf16, #host_to_dram_with_vgm_dst>) -> tensor<2x2x32x32xbf16, #host_to_dram_with_vgm_dst> {
  // CHECK-LABEL: @dram_to_dram_vgm_change
  // CHECK: %[[DRAM:.*]] = d2m.empty() {virtualGridForwardMapping = #map{{[0-9]+}}, virtualGridInverseMapping = #map{{[0-9]+}}} : tensor<2x2x32x32xbf16, #layout{{[0-9]+}}>
  // CHECK: %[[L1:.*]] = d2m.empty() : tensor<2x2x32x32xbf16, #layout{{[0-9]+}}>
  // CHECK: %[[TO_L1:.*]] = d2m.generic
  // CHECK-NEXT: ins(%arg0 : tensor<2x2x32x32xbf16, #layout{{[0-9]+}}>)
  // CHECK-NEXT: outs(%[[L1]] : tensor<2x2x32x32xbf16, #layout{{[0-9]+}}>)
  // CHECK: %[[TO_DRAM:.*]] = d2m.generic
  // CHECK-SAME: grid = #ttcore.grid<2x2, virt_to_physical_map = {{.*}}, physical_to_virt_map = {{.*}}>
  // CHECK-NEXT: ins(%[[TO_L1]] : tensor<2x2x32x32xbf16, #layout{{[0-9]+}}>)
  // CHECK-NEXT: outs(%[[DRAM]] : tensor<2x2x32x32xbf16, #layout{{[0-9]+}}>)
  // CHECK: return %[[TO_DRAM]]
  %0 = d2m.empty() {virtualGridForwardMapping = affine_map<(d0, d1, d2, d3) -> (d0 + 2, d1 + 3, d2, d3)>, virtualGridInverseMapping = affine_map<(d0, d1) -> (0, d0 - 2, d1 - 3)>} : tensor<2x2x32x32xbf16, #host_to_dram_with_vgm_dst>
  %1 = d2m.to_layout %arg0, %0 : tensor<2x2x32x32xbf16, #host_to_dram_with_vgm_dst> into tensor<2x2x32x32xbf16, #host_to_dram_with_vgm_dst> -> tensor<2x2x32x32xbf16, #host_to_dram_with_vgm_dst>
  return %1 : tensor<2x2x32x32xbf16, #host_to_dram_with_vgm_dst>
}

#l1_rank8_vgm_src = #ttcore.metal_layout<logical_shape = 1x8x512x128, dim_alignments = 1x1x256x32, collapsed_intervals = dense<> : tensor<0x2xi64>, l1, sharded>
#dram_rank4_collapsed_dst = #ttcore.metal_layout<logical_shape = 1x8x512x128, dim_alignments = 256x1x32x32, collapsed_intervals = dense<[[0, 3], [3, 4]]> : tensor<2x2xi64>, dram, sharded>

func.func @l1_vgm_to_collapsed_dram_uses_grid_rank(%arg0: tensor<1x1x16x4x1x8x1x1x!ttcore.tile<32x32, bf16>, #l1_rank8_vgm_src>) -> tensor<8x4x16x1x!ttcore.tile<32x32, bf16>, #dram_rank4_collapsed_dst> {
  // CHECK-LABEL: @l1_vgm_to_collapsed_dram_uses_grid_rank
  // CHECK: d2m.generic
  // CHECK-SAME: grid = #ttcore.grid<1x1x16x4, virt_to_physical_map = {{.*}}, physical_to_virt_map = {{.*}}>
  // CHECK: d2m.block_index(3)
  // CHECK: d2m.remote_load {{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}]
  // CHECK: d2m.remote_store {{.*}}[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}]
  %0 = d2m.empty() {virtualGridForwardMapping = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (((d3 + d2 * 4) floordiv 8) mod 8, (d3 + d2 * 4) mod 8, d4, d5, d6, d7)>, virtualGridInverseMapping = affine_map<(d0, d1) -> (0, 0, 0, (d1 floordiv 4 + d0 * 2) mod 16, d1 mod 4)>} : tensor<1x1x16x4x1x8x1x1x!ttcore.tile<32x32, bf16>, #l1_rank8_vgm_src>
  %1 = d2m.to_layout %arg0, %0 : tensor<1x1x16x4x1x8x1x1x!ttcore.tile<32x32, bf16>, #l1_rank8_vgm_src> into tensor<1x1x16x4x1x8x1x1x!ttcore.tile<32x32, bf16>, #l1_rank8_vgm_src> -> tensor<1x1x16x4x1x8x1x1x!ttcore.tile<32x32, bf16>, #l1_rank8_vgm_src>
  %2 = d2m.empty() : tensor<8x4x16x1x!ttcore.tile<32x32, bf16>, #dram_rank4_collapsed_dst>
  %3 = d2m.to_layout %1, %2 : tensor<1x1x16x4x1x8x1x1x!ttcore.tile<32x32, bf16>, #l1_rank8_vgm_src> into tensor<8x4x16x1x!ttcore.tile<32x32, bf16>, #dram_rank4_collapsed_dst> -> tensor<8x4x16x1x!ttcore.tile<32x32, bf16>, #dram_rank4_collapsed_dst>
  return %3 : tensor<8x4x16x1x!ttcore.tile<32x32, bf16>, #dram_rank4_collapsed_dst>
}
