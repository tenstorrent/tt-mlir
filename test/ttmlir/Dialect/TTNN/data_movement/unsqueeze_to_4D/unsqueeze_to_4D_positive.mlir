// RUN: ttmlir-opt %s | FileCheck %s
// Test positive cases for ttnn.unsqueeze_to_4D operation.

module {
  // CHECK-LABEL: func.func @unsqueeze_2d_to_4d
  func.func @unsqueeze_2d_to_4d(%arg0: tensor<32x128xbf16>) -> tensor<1x1x32x128xbf16> {
    // CHECK: %[[C:.*]] = "ttnn.unsqueeze_to_4D"(%arg0) : (tensor<32x128xbf16>) -> tensor<1x1x32x128xbf16>
    %0 = "ttnn.unsqueeze_to_4D"(%arg0) : (tensor<32x128xbf16>) -> tensor<1x1x32x128xbf16>
    // CHECK: return %[[C]] : tensor<1x1x32x128xbf16>
    return %0 : tensor<1x1x32x128xbf16>
  }

  // CHECK-LABEL: func.func @unsqueeze_3d_to_4d
  func.func @unsqueeze_3d_to_4d(%arg0: tensor<8x32x128xbf16>) -> tensor<1x8x32x128xbf16> {
    // CHECK: %[[C:.*]] = "ttnn.unsqueeze_to_4D"(%arg0) : (tensor<8x32x128xbf16>) -> tensor<1x8x32x128xbf16>
    %0 = "ttnn.unsqueeze_to_4D"(%arg0) : (tensor<8x32x128xbf16>) -> tensor<1x8x32x128xbf16>
    // CHECK: return %[[C]] : tensor<1x8x32x128xbf16>
    return %0 : tensor<1x8x32x128xbf16>
  }

  // CHECK-LABEL: func.func @unsqueeze_1d_to_4d
  func.func @unsqueeze_1d_to_4d(%arg0: tensor<128xbf16>) -> tensor<1x1x1x128xbf16> {
    // CHECK: %[[C:.*]] = "ttnn.unsqueeze_to_4D"(%arg0) : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    %0 = "ttnn.unsqueeze_to_4D"(%arg0) : (tensor<128xbf16>) -> tensor<1x1x1x128xbf16>
    // CHECK: return %[[C]] : tensor<1x1x1x128xbf16>
    return %0 : tensor<1x1x1x128xbf16>
  }

  // CHECK-LABEL: func.func @unsqueeze_4d_to_4d
  func.func @unsqueeze_4d_to_4d(%arg0: tensor<2x8x32x128xbf16>) -> tensor<2x8x32x128xbf16> {
    // CHECK: %[[C:.*]] = "ttnn.unsqueeze_to_4D"(%arg0) : (tensor<2x8x32x128xbf16>) -> tensor<2x8x32x128xbf16>
    %0 = "ttnn.unsqueeze_to_4D"(%arg0) : (tensor<2x8x32x128xbf16>) -> tensor<2x8x32x128xbf16>
    // CHECK: return %[[C]] : tensor<2x8x32x128xbf16>
    return %0 : tensor<2x8x32x128xbf16>
  }
}