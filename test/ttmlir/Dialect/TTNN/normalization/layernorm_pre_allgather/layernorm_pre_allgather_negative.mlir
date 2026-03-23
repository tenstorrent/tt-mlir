// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for ttnn layernorm_pre_allgather op verifier

// -----

// Verify that output last dimension must be 64.
module attributes {} {
  func.func @layernorm_pre_allgather_wrong_last_dim(%arg0: tensor<1x1x32x1024xbf16>, %arg1: tensor<1x1x32x32xbf16>) -> tensor<1x1x32x128xbf16> {
    // CHECK: error: 'ttnn.layernorm_pre_allgather' op output last dimension must be 64
    %0 = "ttnn.layernorm_pre_allgather"(%arg0, %arg1) : (tensor<1x1x32x1024xbf16>, tensor<1x1x32x32xbf16>) -> tensor<1x1x32x128xbf16>
    return %0 : tensor<1x1x32x128xbf16>
  }
}

// -----

// Verify that output shape must match input shape except last dim.
module attributes {} {
  func.func @layernorm_pre_allgather_shape_mismatch(%arg0: tensor<1x1x32x1024xbf16>, %arg1: tensor<1x1x32x32xbf16>) -> tensor<1x1x64x64xbf16> {
    // CHECK: error: 'ttnn.layernorm_pre_allgather' op output shape must match input shape except for the last dimension
    %0 = "ttnn.layernorm_pre_allgather"(%arg0, %arg1) : (tensor<1x1x32x1024xbf16>, tensor<1x1x32x32xbf16>) -> tensor<1x1x64x64xbf16>
    return %0 : tensor<1x1x64x64xbf16>
  }
}

// -----

// Verify that output rank must match input rank.
module attributes {} {
  func.func @layernorm_pre_allgather_rank_mismatch(%arg0: tensor<1x1x32x1024xbf16>, %arg1: tensor<1x1x32x32xbf16>) -> tensor<1x32x64xbf16> {
    // CHECK: error: 'ttnn.layernorm_pre_allgather' op output rank must match input rank
    %0 = "ttnn.layernorm_pre_allgather"(%arg0, %arg1) : (tensor<1x1x32x1024xbf16>, tensor<1x1x32x32xbf16>) -> tensor<1x32x64xbf16>
    return %0 : tensor<1x32x64xbf16>
  }
}
