// RUN: not ttmlir-opt -split-input-file %s 2>&1 | FileCheck %s

// Verifier rejections for ttir.cross_entropy_bw. Rank is deliberately not
// constrained here; TTIRToTTIRDecomposition normalizes to the kernel's
// (N, 1, H, W) / (N, H) / (1, 1, 1, 1) form.

// CHECK: error: 'ttir.cross_entropy_bw' op target batch extent (2) must match input batch extent (4)
module {
  func.func @batch_mismatch(%input: tensor<4x1x32x64xbf16>, %target: tensor<2x32xui32>, %grad: tensor<1x1x1x1xbf16>)
      -> tensor<4x1x32x64xbf16> {
    %0 = "ttir.cross_entropy_bw"(%input, %target, %grad) <{scaler = 3.125e-02 : f32}>
        : (tensor<4x1x32x64xbf16>, tensor<2x32xui32>, tensor<1x1x1x1xbf16>) -> tensor<4x1x32x64xbf16>
    return %0 : tensor<4x1x32x64xbf16>
  }
}

// -----

// CHECK: error: 'ttir.cross_entropy_bw' op target last dimension (16) must match input dimension -2 (32)
module {
  func.func @rows_mismatch(%input: tensor<4x1x32x64xbf16>, %target: tensor<4x16xui32>, %grad: tensor<1x1x1x1xbf16>)
      -> tensor<4x1x32x64xbf16> {
    %0 = "ttir.cross_entropy_bw"(%input, %target, %grad) <{scaler = 3.125e-02 : f32}>
        : (tensor<4x1x32x64xbf16>, tensor<4x16xui32>, tensor<1x1x1x1xbf16>) -> tensor<4x1x32x64xbf16>
    return %0 : tensor<4x1x32x64xbf16>
  }
}

// -----

// CHECK: error: 'ttir.cross_entropy_bw' op input must have rank at least 2 (..., H, W), got rank 1
module {
  func.func @input_rank_too_low(%input: tensor<64xbf16>, %target: tensor<1xui32>, %grad: tensor<1x1x1x1xbf16>)
      -> tensor<64xbf16> {
    %0 = "ttir.cross_entropy_bw"(%input, %target, %grad) <{scaler = 3.125e-02 : f32}>
        : (tensor<64xbf16>, tensor<1xui32>, tensor<1x1x1x1xbf16>) -> tensor<64xbf16>
    return %0 : tensor<64xbf16>
  }
}

// -----

// ttml only supports a scalar gradient.
// CHECK: error: 'ttir.cross_entropy_bw' op grad must have all dimensions equal to 1, got 4, 1, 32, 64
module {
  func.func @grad_not_scalar(%input: tensor<4x1x32x64xbf16>, %target: tensor<4x32xui32>, %grad: tensor<4x1x32x64xbf16>)
      -> tensor<4x1x32x64xbf16> {
    %0 = "ttir.cross_entropy_bw"(%input, %target, %grad) <{scaler = 3.125e-02 : f32}>
        : (tensor<4x1x32x64xbf16>, tensor<4x32xui32>, tensor<4x1x32x64xbf16>) -> tensor<4x1x32x64xbf16>
    return %0 : tensor<4x1x32x64xbf16>
  }
}

// -----

// CHECK: error: 'ttir.cross_entropy_bw' op result shape must match input shape
module {
  func.func @result_reduced_like_fw(%input: tensor<4x1x32x64xbf16>, %target: tensor<4x32xui32>, %grad: tensor<1x1x1x1xbf16>)
      -> tensor<4x1x32x1xbf16> {
    %0 = "ttir.cross_entropy_bw"(%input, %target, %grad) <{scaler = 3.125e-02 : f32}>
        : (tensor<4x1x32x64xbf16>, tensor<4x32xui32>, tensor<1x1x1x1xbf16>) -> tensor<4x1x32x1xbf16>
    return %0 : tensor<4x1x32x1xbf16>
  }
}

// -----

// CHECK: error: 'ttir.cross_entropy_bw' op target must have an integer element type, got 'f32'
module {
  func.func @target_not_integer(%input: tensor<4x1x32x64xbf16>, %target: tensor<4x32xf32>, %grad: tensor<1x1x1x1xbf16>)
      -> tensor<4x1x32x64xbf16> {
    %0 = "ttir.cross_entropy_bw"(%input, %target, %grad) <{scaler = 3.125e-02 : f32}>
        : (tensor<4x1x32x64xbf16>, tensor<4x32xf32>, tensor<1x1x1x1xbf16>) -> tensor<4x1x32x64xbf16>
    return %0 : tensor<4x1x32x64xbf16>
  }
}
