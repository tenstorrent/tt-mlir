// RUN: not ttmlir-opt -split-input-file %s 2>&1 | FileCheck %s

// Verifier rejections for ttnn.cross_entropy_fw.

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

// ttml::metal::cross_entropy_fw hardcodes its circular buffers to Float16_b.
// CHECK: error: 'ttnn.cross_entropy_fw' op input must have bf16 dtype, got f32
module {
  func.func @input_not_bf16(%input: tensor<4x1x32x64xf32>, %target: tensor<4x32xui32>)
      -> tensor<4x1x32x1xf32> {
    %0 = "ttnn.cross_entropy_fw"(%input, %target)
        : (tensor<4x1x32x64xf32>, tensor<4x32xui32>) -> tensor<4x1x32x1xf32>
    return %0 : tensor<4x1x32x1xf32>
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
