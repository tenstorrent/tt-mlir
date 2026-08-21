// RUN: not ttmlir-opt %s 2>&1 | FileCheck %s

module {
  // CHECK: error: 'ttnn.rms_norm_post_all_gather' op input and output must have the same shape
  func.func @bad_output_shape(%arg0: tensor<1x1x32x128xbf16>, %arg1: tensor<1x1x32x64xbf16>) -> tensor<1x1x32x64xbf16> {
    %0 = "ttnn.rms_norm_post_all_gather"(%arg0, %arg1) <{epsilon = 1.000000e-12 : f32, operandSegmentSizes = array<i32: 1, 1, 0, 0>}> : (tensor<1x1x32x128xbf16>, tensor<1x1x32x64xbf16>) -> tensor<1x1x32x64xbf16>
    return %0 : tensor<1x1x32x64xbf16>
  }
}
