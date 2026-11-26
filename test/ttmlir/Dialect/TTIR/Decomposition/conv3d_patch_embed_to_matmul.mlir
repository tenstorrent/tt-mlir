// RUN: ttmlir-opt --ttir-to-ttir-decomposition %s | FileCheck %s

// A patchify conv3d (kernel spans the whole spatial input, stride == kernel, no
// padding, groups == 1) is exactly a matmul, so it is rewritten away from the
// conv kernel. This is the Qwen2.5-VL vision patch-embed shape (tt-xla #1662).
module {
  // CHECK-LABEL: func.func @conv3d_patch_embed
  func.func @conv3d_patch_embed(%input: tensor<2x3x2x4x4xf32>, %weight: tensor<8x3x2x4x4xf32>) -> tensor<2x8x1x1x1xf32> {
    // CHECK-NOT: ttir.conv3d
    // CHECK: ttir.matmul
    // CHECK-SAME: transpose_b = true
    %0 = "ttir.conv3d"(%input, %weight) <{batch_dim = 0 : i64, channel_dim = 1 : i64, depth_dim = 2 : i64, groups = 1 : i32, height_dim = 3 : i64, padding = array<i32: 0, 0, 0>, padding_mode = "zeros", stride = array<i32: 2, 4, 4>, width_dim = 4 : i64}> : (tensor<2x3x2x4x4xf32>, tensor<8x3x2x4x4xf32>) -> tensor<2x8x1x1x1xf32>
    return %0 : tensor<2x8x1x1x1xf32>
  }

  // A real conv3d whose kernel does NOT cover the full spatial extent must stay
  // a convolution (here already NDHWC so the channel-last pass leaves it alone).
  // CHECK-LABEL: func.func @conv3d_real_stays_conv
  func.func @conv3d_real_stays_conv(%input: tensor<2x4x8x8x3xf32>, %weight: tensor<8x3x2x4x4xf32>) -> tensor<2x2x2x2x8xf32> {
    // CHECK-NOT: ttir.matmul
    // CHECK: ttir.conv3d
    %0 = "ttir.conv3d"(%input, %weight) <{batch_dim = 0 : i64, channel_dim = 4 : i64, depth_dim = 1 : i64, groups = 1 : i32, height_dim = 2 : i64, padding = array<i32: 0, 0, 0>, padding_mode = "zeros", stride = array<i32: 2, 4, 4>, width_dim = 3 : i64}> : (tensor<2x4x8x8x3xf32>, tensor<8x3x2x4x4xf32>) -> tensor<2x2x2x2x8xf32>
    return %0 : tensor<2x2x2x2x8xf32>
  }
}
