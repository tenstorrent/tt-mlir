// RUN: not ttmlir-opt -split-input-file %s 2>&1 | FileCheck %s

// CHECK: error: 'ttnn.cross_entropy_fw' op input dim 1 must be 1, got 2
module {
  func.func @channel_dim_not_one(%input: tensor<4x2x32x64xbf16>, %target: tensor<4x32xui32>)
      -> tensor<4x2x32x1xbf16> {
    %0 = "ttnn.cross_entropy_fw"(%input, %target)
        : (tensor<4x2x32x64xbf16>, tensor<4x32xui32>) -> tensor<4x2x32x1xbf16>
    return %0 : tensor<4x2x32x1xbf16>
  }
}

// -----

// CHECK: error: 'ttnn.cross_entropy_fw' op input must be a 4D tensor (N, 1, H, W), got rank 2
module {
  func.func @input_not_4d(%input: tensor<32x64xbf16>, %target: tensor<32xui32>)
      -> tensor<32x1xbf16> {
    %0 = "ttnn.cross_entropy_fw"(%input, %target)
        : (tensor<32x64xbf16>, tensor<32xui32>) -> tensor<32x1xbf16>
    return %0 : tensor<32x1xbf16>
  }
}

// -----

// CHECK: error: 'ttnn.cross_entropy_fw' op target must be a 2D tensor (N, H), got rank 1
module {
  func.func @target_not_2d(%input: tensor<1x1x32x64xbf16>, %target: tensor<32xui32>)
      -> tensor<1x1x32x1xbf16> {
    %0 = "ttnn.cross_entropy_fw"(%input, %target)
        : (tensor<1x1x32x64xbf16>, tensor<32xui32>) -> tensor<1x1x32x1xbf16>
    return %0 : tensor<1x1x32x1xbf16>
  }
}

// -----

// CHECK: error: 'ttnn.cross_entropy_fw' op result shape must be input shape with the last dimension set to 1
module {
  func.func @bad_result_shape(%input: tensor<4x1x32x64xbf16>, %target: tensor<4x32xui32>)
      -> tensor<4x1x32x64xbf16> {
    %0 = "ttnn.cross_entropy_fw"(%input, %target)
        : (tensor<4x1x32x64xbf16>, tensor<4x32xui32>) -> tensor<4x1x32x64xbf16>
    return %0 : tensor<4x1x32x64xbf16>
  }
}
