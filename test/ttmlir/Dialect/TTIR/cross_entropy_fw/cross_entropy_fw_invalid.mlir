// RUN: not ttmlir-opt -split-input-file %s 2>&1 | FileCheck %s

// Verifier rejections for ttir.cross_entropy_fw.

// CHECK: error: 'ttir.cross_entropy_fw' op input dim 1 must be 1, got 2
module {
  func.func @channel_dim_not_one(%input: tensor<4x2x32x64xbf16>, %target: tensor<4x32xui32>)
      -> tensor<4x2x32x1xbf16> {
    %0 = "ttir.cross_entropy_fw"(%input, %target)
        : (tensor<4x2x32x64xbf16>, tensor<4x32xui32>) -> tensor<4x2x32x1xbf16>
    return %0 : tensor<4x2x32x1xbf16>
  }
}

// -----

// CHECK: error: 'ttir.cross_entropy_fw' op input must be a 4D tensor (N, 1, H, W), got rank 3
module {
  func.func @input_not_4d(%input: tensor<1x32x64xbf16>, %target: tensor<1x32xui32>)
      -> tensor<1x32x1xbf16> {
    %0 = "ttir.cross_entropy_fw"(%input, %target)
        : (tensor<1x32x64xbf16>, tensor<1x32xui32>) -> tensor<1x32x1xbf16>
    return %0 : tensor<1x32x1xbf16>
  }
}

// -----

// CHECK: error: 'ttir.cross_entropy_fw' op target dim 1 (16) must match input dim 2 (32)
module {
  func.func @target_rows_mismatch(%input: tensor<4x1x32x64xbf16>, %target: tensor<4x16xui32>)
      -> tensor<4x1x32x1xbf16> {
    %0 = "ttir.cross_entropy_fw"(%input, %target)
        : (tensor<4x1x32x64xbf16>, tensor<4x16xui32>) -> tensor<4x1x32x1xbf16>
    return %0 : tensor<4x1x32x1xbf16>
  }
}

// -----

// CHECK: error: 'ttir.cross_entropy_fw' op target must have an integer element type, got 'f32'
module {
  func.func @target_not_integer(%input: tensor<4x1x32x64xbf16>, %target: tensor<4x32xf32>)
      -> tensor<4x1x32x1xbf16> {
    %0 = "ttir.cross_entropy_fw"(%input, %target)
        : (tensor<4x1x32x64xbf16>, tensor<4x32xf32>) -> tensor<4x1x32x1xbf16>
    return %0 : tensor<4x1x32x1xbf16>
  }
}
