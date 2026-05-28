// Wan VAE Conv3d fragment — four kernel shapes that exercise the
// representative Conv3d configurations in the Wan VAE model.
//
//   (3,3,3) — main residual block convs.
//   (3,1,1) — temporal-only convs (e.g. time_conv).
//   (1,1,1) — pointwise channel mixing.
//   (1,2,2) — patchify head (stride == kernel).
//
// Each function is independent so the perf gate measures the optimizer's
// pick per shape rather than a fused chain.
module {
  func.func @conv3d_3x3x3(
      %arg0: tensor<1x8x28x28x128xbf16>,
      %arg1: tensor<32x128x3x3x3xbf16>)
      -> tensor<1x6x26x26x32xbf16> {
    %0 = "ttir.conv3d"(%arg0, %arg1) <{
        stride = array<i32: 1, 1, 1>,
        padding = array<i32: 0, 0, 0>,
        groups = 1 : i32,
        padding_mode = "zeros"
      }> : (tensor<1x8x28x28x128xbf16>, tensor<32x128x3x3x3xbf16>)
        -> tensor<1x6x26x26x32xbf16>
    return %0 : tensor<1x6x26x26x32xbf16>
  }

  func.func @conv3d_3x1x1_temporal(
      %arg0: tensor<1x8x16x16x96xbf16>,
      %arg1: tensor<96x96x3x1x1xbf16>)
      -> tensor<1x6x16x16x96xbf16> {
    %0 = "ttir.conv3d"(%arg0, %arg1) <{
        stride = array<i32: 1, 1, 1>,
        padding = array<i32: 0, 0, 0>,
        groups = 1 : i32,
        padding_mode = "zeros"
      }> : (tensor<1x8x16x16x96xbf16>, tensor<96x96x3x1x1xbf16>)
        -> tensor<1x6x16x16x96xbf16>
    return %0 : tensor<1x6x16x16x96xbf16>
  }

  func.func @conv3d_1x1x1_pointwise(
      %arg0: tensor<1x4x16x16x192xbf16>,
      %arg1: tensor<96x192x1x1x1xbf16>)
      -> tensor<1x4x16x16x96xbf16> {
    %0 = "ttir.conv3d"(%arg0, %arg1) <{
        stride = array<i32: 1, 1, 1>,
        padding = array<i32: 0, 0, 0>,
        groups = 1 : i32,
        padding_mode = "zeros"
      }> : (tensor<1x4x16x16x192xbf16>, tensor<96x192x1x1x1xbf16>)
        -> tensor<1x4x16x16x96xbf16>
    return %0 : tensor<1x4x16x16x96xbf16>
  }

  func.func @conv3d_1x2x2_patchify(
      %arg0: tensor<1x4x32x32x32xbf16>,
      %arg1: tensor<96x32x1x2x2xbf16>)
      -> tensor<1x4x16x16x96xbf16> {
    %0 = "ttir.conv3d"(%arg0, %arg1) <{
        stride = array<i32: 1, 2, 2>,
        padding = array<i32: 0, 0, 0>,
        groups = 1 : i32,
        padding_mode = "zeros"
      }> : (tensor<1x4x32x32x32xbf16>, tensor<96x32x1x2x2xbf16>)
        -> tensor<1x4x16x16x96xbf16>
    return %0 : tensor<1x4x16x16x96xbf16>
  }
}
