// RUN: ttmlir-opt --ttcore-register-device --d2m-lower-to-layout -o %t %s
// RUN: FileCheck %s --input-file=%t

#layout = #ttcore.metal_layout<logical_shape = 1024x1024, dim_alignments = 256x256, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1>
#layout2 = #ttcore.metal_layout<logical_shape = 256x768, dim_alignments = 32x256, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1>

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
  // Reblocking should use view_layout and datamovement threads since it's just redistributing already tiled data
  // CHECK: d2m.view_layout %arg0 : tensor<1x1x8x24x!ttcore.tile<32x32, f32>, #layout{{[0-9]*}}> -> tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout{{[0-9]*}}>
  // CHECK: d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<8x8>
  // CHECK-SAME: threads = [#d2m.thread<datamovement>]

  %1 = d2m.to_layout %arg0, %0 : tensor<1x1x8x24x!ttcore.tile<32x32, f32>, #layout2> into tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout2>
    -> tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout2>

  return %1 : tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout2>
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
