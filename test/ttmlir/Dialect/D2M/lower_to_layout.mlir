#layout = #ttcore.metal_layout<logical_shape = 1024x1024, dim_alignments = 256x256, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1,, index_map = map(0)>
#layout1 = #ttcore.metal_layout<logical_shape = 256x768, dim_alignments = 32x256, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1,, index_map = map(0)>
module {
  func.func @tilize(%arg0: tensor<1024x1024xf32>) -> tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout> {
    %0 = d2m.empty() : tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout>
    %1 = d2m.to_layout %arg0, %0 : tensor<1024x1024xf32> into tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout> -> tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout>
    return %1 : tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout>
  }
  func.func @reblock(%arg0: tensor<1x1x8x24x!ttcore.tile<32x32, f32>, #layout1>) -> tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout1> {
    %0 = d2m.empty() : tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout1>
    %1 = d2m.to_layout %arg0, %0 : tensor<1x1x8x24x!ttcore.tile<32x32, f32>, #layout1> into tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout1> -> tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout1>
    return %1 : tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout1>
  }
  func.func @untilize(%arg0: tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout>) -> tensor<1024x1024xf32> {
    %0 = d2m.empty() : tensor<1024x1024xf32>
    %1 = d2m.to_layout %arg0, %0 : tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout> into tensor<1024x1024xf32> -> tensor<1024x1024xf32>
    return %1 : tensor<1024x1024xf32>
  }
  func.func @compound(%arg0: tensor<256x768xf32>) -> tensor<256x768xf32> {
    %0 = d2m.empty() : tensor<8x8x1x3x!ttcore.tile<32x32, f32>, #layout1>
    %1 = d2m.empty() : tensor<256x768xf32>
    return %arg0 : tensor<256x768xf32>
  }
  func.func @old_behavior_example(%arg0: tensor<1024x1024xf32>) -> tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout> {
    %0 = d2m.empty() : tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout>
    %1 = d2m.to_layout %arg0, %0 : tensor<1024x1024xf32> into tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout> -> tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout>
    return %1 : tensor<8x8x4x4x!ttcore.tile<32x32, f32>, #layout>
  }
}
