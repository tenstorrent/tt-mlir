// RUN: ttmlir-opt --ttcore-register-device --ttir-lower-to-layout -o %t %s
// RUN: FileCheck %s --input-file=%t

#layout = #ttcore.metal_layout<logical_shape = 1024x1024, dim_alignments = 256x256, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1>
#layout2 = #ttcore.metal_layout<logical_shape = 256x768, dim_alignments = 32x256, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1>

// Test that verifies the NEW behavior: distribute to 8x8 grid first, then tilize
func.func @tilize(%arg0: tensor<1024x1024xf32>) -> tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout> {
  %0 = ttir.empty() : tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout>

  // CHECK-LABEL: @tilize
  // Verify the operation creates intermediate 8x8 distributed tensor
  // CHECK: %[[TILED:.*]] = ttir.empty() : tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout>
  // CHECK: %[[INTERMEDIATE:.*]] = ttir.empty() : tensor<8x8x128x128xf32, #layout>
  // CHECK: %[[TO_DEVICE:.*]] = ttir.to_layout %arg0, %[[INTERMEDIATE]] : tensor<1024x1024xf32> into tensor<8x8x128x128xf32, #layout>
  // CHECK: %[[RESULT:.*]] = ttir.generic {block_factors = [1, 1], grid = #ttcore.grid<8x8>
  // CHECK-SAME: threads = [#ttir.thread<compute>]
  // CHECK-NEXT: ins(%[[TO_DEVICE]] : tensor<8x8x128x128xf32, #layout>)
  // CHECK-NEXT: outs(%[[TILED]] : tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout>)
  // CHECK: ttir.tile_tilize_block
  // CHECK-NOT: ttir.view_layout
  // CHECK-NOT: threads = [#ttir.thread<datamovement>]

  %1 = ttir.to_layout %arg0, %0 : tensor<1024x1024xf32> into tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout>
    -> tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout>

  return %1 : tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout>
}

// Test reblock operation - redistributing already tiled data
func.func @reblock(%arg0: tensor<1x1x8x24x!ttcore.tile<32x32, f32>, #layout2>) -> tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout2> {
  %0 = ttir.empty() : tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout2>

  // CHECK-LABEL: @reblock
  // Reblocking should use view_layout and datamovement threads since it's just redistributing already tiled data
  // CHECK: ttir.view_layout %arg0 : tensor<1x1x8x24x!ttcore.tile<32x32, f32>, #layout{{[0-9]*}}> -> tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout{{[0-9]*}}>
  // CHECK: ttir.generic {block_factors = [1, 1], grid = #ttcore.grid<8x8>
  // CHECK-SAME: threads = [#ttir.thread<datamovement>]

  %1 = ttir.to_layout %arg0, %0 : tensor<1x1x8x24x!ttcore.tile<32x32, f32>, #layout2> into tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout2>
    -> tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout2>

  return %1 : tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout2>
}

// Test untilize with 8x8 distribution
func.func @untilize(%arg0: tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout>) -> tensor<1024x1024xf32> {
  %0 = ttir.empty() : tensor<1024x1024xf32>

  // CHECK-LABEL: @untilize
  // CHECK: %[[HOST:.*]] = ttir.empty() : tensor<1024x1024xf32>
  // CHECK: %[[INTERMEDIATE:.*]] = ttir.empty() : tensor<8x8x128x128xf32, #layout>
  // CHECK: %[[UNTILIZED:.*]] = ttir.generic {block_factors = [1, 1], grid = #ttcore.grid<8x8>
  // CHECK-SAME: threads = [#ttir.thread<compute>]
  // CHECK: ttir.tile_untilize_block
  // CHECK: ttir.to_layout %[[UNTILIZED]], %[[HOST]] : tensor<8x8x128x128xf32, #layout> into tensor<1024x1024xf32>

  %1 = ttir.to_layout %arg0, %0 : tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout> into tensor<1024x1024xf32>
    -> tensor<1024x1024xf32>

  return %1 : tensor<1024x1024xf32>
}

// Test compound operation with 8x8 distribution
func.func @compound(%arg0: tensor<256x768xf32>) -> tensor<256x768xf32> {
  %0 = ttir.empty() : tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout2>
  %1 = ttir.empty() : tensor<256x768xf32>

  // CHECK-LABEL: @compound
  // CHECK: ttir.empty() : tensor<8x8x32x96xf32, #layout{{[0-9]*}}>
  // CHECK: ttir.to_layout %arg0, %{{.*}} : tensor<256x768xf32> into tensor<8x8x32x96xf32, #layout{{[0-9]*}}>
  // CHECK: ttir.generic {{{.*}}grid = #ttcore.grid<8x8>{{.*}}threads = [#ttir.thread<compute>]
  // CHECK: ttir.tile_tilize_block
  // CHECK: ttir.generic {{{.*}}grid = #ttcore.grid<8x8>{{.*}}threads = [#ttir.thread<compute>]
  // CHECK: ttir.tile_untilize_block
  // CHECK: ttir.to_layout %{{.*}} : tensor<8x8x32x96xf32, #layout{{[0-9]*}}> into tensor<256x768xf32>

  %2 = ttir.to_layout %arg0, %0 : tensor<256x768xf32> into tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout2>
    -> tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout2>

  %3 = ttir.to_layout %2, %1 : tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout2> into tensor<256x768xf32>
    -> tensor<256x768xf32>

  return %3 : tensor<256x768xf32>
}

// Test case showing what the OLD behavior would look like (for documentation purposes)
// This is what we're moving AWAY from
func.func @old_behavior_example(%arg0: tensor<1024x1024xf32>) -> tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout> {
  %0 = ttir.empty() : tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout>

  // CHECK-LABEL: @old_behavior_example
  // The OLD behavior that we DON'T want would have:
  // 1. Move to 1x1 core: tensor<1024x1024xf32> -> tensor<1x1x1024x1024xf32>
  // 2. Tilize on 1x1: tensor<1x1x1024x1024xf32> -> tensor<1x1x32x32x!ttcore.tile<32x32, f32>>
  // 3. View as 8x8: tensor<1x1x32x32x!ttcore.tile<32x32, f32>> -> tensor<8x8x4x4x!ttcore.tile<32x32, f32>>
  // 4. Distribute with datamovement threads

  // But the new implementation should produce the same output as distributed_tilize:
  // CHECK: ttir.empty() : tensor<8x8x128x128xf32, #layout>
  // CHECK: ttir.to_layout %arg0, %{{.*}} : tensor<1024x1024xf32> into tensor<8x8x128x128xf32, #layout>
  // CHECK: ttir.generic {{{.*}}grid = #ttcore.grid<8x8>{{.*}}threads = [#ttir.thread<compute>]

  %1 = ttir.to_layout %arg0, %0 : tensor<1024x1024xf32> into tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout>
    -> tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout>

  return %1 : tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout>
}
