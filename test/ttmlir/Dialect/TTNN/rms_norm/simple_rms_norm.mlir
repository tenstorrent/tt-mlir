// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module {
  func.func @forward(%arg0: tensor<512x1024xbf16>) -> tensor<512x1024xbf16> {
    %0 = ttir.empty() : tensor<512x1024xbf16>
    // CHECK: "ttnn.rms_norm"
    // Test basic RMS norm with normalization over last dimension [1024]
    %1 = "ttir.rms_norm"(%arg0, %0) <{normalized_shape = array<i64: 1024>, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 1, 0, 0>}> : (tensor<512x1024xbf16>, tensor<512x1024xbf16>) -> tensor<512x1024xbf16>
    return %1 : tensor<512x1024xbf16>
  }

  func.func @forward_with_weight(%arg0: tensor<512x1024xbf16>) -> tensor<512x1024xbf16> {
    %0 = ttir.empty() : tensor<512x1024xbf16>
    %1 = ttir.empty() : tensor<1024xbf16>
    // CHECK: "ttnn.rms_norm"
    // Test RMS norm with weight
    %2 = "ttir.rms_norm"(%arg0, %0, %1) <{normalized_shape = array<i64: 1024>, epsilon = 1.000000e-12 : f32, operandSegmentSizes = array<i32: 1, 1, 1, 0>}> : (tensor<512x1024xbf16>, tensor<512x1024xbf16>, tensor<1024xbf16>) -> tensor<512x1024xbf16>
    return %2 : tensor<512x1024xbf16>
  }

  func.func @forward_with_weight_and_bias(%arg0: tensor<512x1024xbf16>) -> tensor<512x1024xbf16> {
    %0 = ttir.empty() : tensor<512x1024xbf16>
    %1 = ttir.empty() : tensor<1024xbf16>
    %2 = ttir.empty() : tensor<1024xbf16>
    // CHECK: "ttnn.rms_norm"
    // Test RMS norm with weight and bias
    %3 = "ttir.rms_norm"(%arg0, %0, %1, %2) <{normalized_shape = array<i64: 1024>, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 1, 1, 1>}> : (tensor<512x1024xbf16>, tensor<512x1024xbf16>, tensor<1024xbf16>, tensor<1024xbf16>) -> tensor<512x1024xbf16>
    return %3 : tensor<512x1024xbf16>
  }
}
