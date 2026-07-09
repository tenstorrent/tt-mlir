// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-ttnn-decomposition-pass=false" -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // Basic fused RMSNorm (no activation, no residual).
  func.func @forward(%arg0: tensor<512x1024xbf16>) -> tensor<512x1024xbf16> {
    // CHECK: "ttnn.dit_rms_norm_unary_fused"
    %1 = "ttir.dit_rms_norm_unary_fused"(%arg0) <{normalized_shape = array<i64: 1024>, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 0, 0, 0>}> : (tensor<512x1024xbf16>) -> tensor<512x1024xbf16>
    return %1 : tensor<512x1024xbf16>
  }

  // Fused RMSNorm with weight and a SiLU activation.
  func.func @forward_with_weight_and_activation(%arg0: tensor<512x1024xbf16>) -> tensor<512x1024xbf16> {
    %1 = ttir.empty() : tensor<1024xbf16>
    // CHECK: "ttnn.dit_rms_norm_unary_fused"
    // CHECK-SAME: activation = "silu"
    %2 = "ttir.dit_rms_norm_unary_fused"(%arg0, %1) <{normalized_shape = array<i64: 1024>, epsilon = 1.000000e-05 : f32, activation = "silu", operandSegmentSizes = array<i32: 1, 1, 0, 0>}> : (tensor<512x1024xbf16>, tensor<1024xbf16>) -> tensor<512x1024xbf16>
    return %2 : tensor<512x1024xbf16>
  }

  // Fused RMSNorm with weight, bias, residual input, and GELU activation.
  func.func @forward_with_residual(%arg0: tensor<512x1024xbf16>, %res: tensor<512x1024xbf16>) -> tensor<512x1024xbf16> {
    %1 = ttir.empty() : tensor<1024xbf16>
    %2 = ttir.empty() : tensor<1024xbf16>
    // CHECK: "ttnn.dit_rms_norm_unary_fused"
    // CHECK-SAME: activation = "gelu"
    %3 = "ttir.dit_rms_norm_unary_fused"(%arg0, %1, %2, %res) <{normalized_shape = array<i64: 1024>, epsilon = 1.000000e-05 : f32, activation = "gelu", operandSegmentSizes = array<i32: 1, 1, 1, 1>}> : (tensor<512x1024xbf16>, tensor<1024xbf16>, tensor<1024xbf16>, tensor<512x1024xbf16>) -> tensor<512x1024xbf16>
    return %3 : tensor<512x1024xbf16>
  }
}
