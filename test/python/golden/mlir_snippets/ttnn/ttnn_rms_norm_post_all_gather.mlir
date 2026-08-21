module attributes {} {
  func.func @rms_norm_post_all_gather_forward(
      %input: tensor<1x1x128x64xbf16>,
      %stats: tensor<1x1x128x32xbf16>,
      %weight: tensor<64xbf16>) -> tensor<1x1x128x64xbf16> {
    %0 = "ttnn.rms_norm_post_all_gather"(%input, %stats, %weight) <{
      epsilon = 1.000000e-12 : f32,
      operandSegmentSizes = array<i32: 1, 1, 1, 0>
    }> : (tensor<1x1x128x64xbf16>, tensor<1x1x128x32xbf16>, tensor<64xbf16>) -> tensor<1x1x128x64xbf16>
    return %0 : tensor<1x1x128x64xbf16>
  }
}
