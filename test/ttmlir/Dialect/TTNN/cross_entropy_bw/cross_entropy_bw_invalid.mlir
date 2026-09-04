// RUN: not ttmlir-opt -split-input-file %s 2>&1 | FileCheck %s

// CHECK: error: 'ttnn.cross_entropy_bw' op input dim 1 must be 1, got 2
module {
  func.func @channel_dim_not_one(%input: tensor<4x2x32x64xbf16>, %target: tensor<4x32xui32>, %grad: tensor<1x1x1x1xbf16>)
      -> tensor<4x2x32x64xbf16> {
    %0 = "ttnn.cross_entropy_bw"(%input, %target, %grad) <{scaler = 3.125e-02 : f32}>
        : (tensor<4x2x32x64xbf16>, tensor<4x32xui32>, tensor<1x1x1x1xbf16>) -> tensor<4x2x32x64xbf16>
    return %0 : tensor<4x2x32x64xbf16>
  }
}

// -----

// ttml only supports a scalar gradient.
// CHECK: error: 'ttnn.cross_entropy_bw' op grad must be a (1, 1, 1, 1) tensor, got 1, 1, 1, 32
module {
  func.func @grad_not_scalar(%input: tensor<4x1x32x64xbf16>, %target: tensor<4x32xui32>, %grad: tensor<1x1x1x32xbf16>)
      -> tensor<4x1x32x64xbf16> {
    %0 = "ttnn.cross_entropy_bw"(%input, %target, %grad) <{scaler = 3.125e-02 : f32}>
        : (tensor<4x1x32x64xbf16>, tensor<4x32xui32>, tensor<1x1x1x32xbf16>) -> tensor<4x1x32x64xbf16>
    return %0 : tensor<4x1x32x64xbf16>
  }
}

// -----

// CHECK: error: 'ttnn.cross_entropy_bw' op input must be a 4D tensor (N, 1, H, W), got rank 2
module {
  func.func @input_not_4d(%input: tensor<32x64xbf16>, %target: tensor<32xui32>, %grad: tensor<1x1x1x1xbf16>)
      -> tensor<32x64xbf16> {
    %0 = "ttnn.cross_entropy_bw"(%input, %target, %grad) <{scaler = 3.125e-02 : f32}>
        : (tensor<32x64xbf16>, tensor<32xui32>, tensor<1x1x1x1xbf16>) -> tensor<32x64xbf16>
    return %0 : tensor<32x64xbf16>
  }
}

// -----

// CHECK: error: 'ttnn.cross_entropy_bw' op target must be a 2D tensor (N, H), got rank 1
module {
  func.func @target_not_2d(%input: tensor<1x1x32x64xbf16>, %target: tensor<32xui32>, %grad: tensor<1x1x1x1xbf16>)
      -> tensor<1x1x32x64xbf16> {
    %0 = "ttnn.cross_entropy_bw"(%input, %target, %grad) <{scaler = 3.125e-02 : f32}>
        : (tensor<1x1x32x64xbf16>, tensor<32xui32>, tensor<1x1x1x1xbf16>) -> tensor<1x1x32x64xbf16>
    return %0 : tensor<1x1x32x64xbf16>
  }
}

// -----

// CHECK: error: 'ttnn.cross_entropy_bw' op result shape must match input shape
module {
  func.func @bad_result_shape(%input: tensor<4x1x32x64xbf16>, %target: tensor<4x32xui32>, %grad: tensor<1x1x1x1xbf16>)
      -> tensor<4x1x32x1xbf16> {
    %0 = "ttnn.cross_entropy_bw"(%input, %target, %grad) <{scaler = 3.125e-02 : f32}>
        : (tensor<4x1x32x64xbf16>, tensor<4x32xui32>, tensor<1x1x1x1xbf16>) -> tensor<4x1x32x1xbf16>
    return %0 : tensor<4x1x32x1xbf16>
  }
}
