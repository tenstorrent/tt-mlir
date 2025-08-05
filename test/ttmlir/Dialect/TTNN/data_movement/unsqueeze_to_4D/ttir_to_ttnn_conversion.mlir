// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
// Test that TTIR unsqueeze operations are correctly converted to ttnn.unsqueeze_to_4D when appropriate.

module {
  // CHECK-LABEL: func.func @unsqueeze_2d_to_4d_conversion
  func.func @unsqueeze_2d_to_4d_conversion(%arg0: tensor<32x128xbf16>) -> tensor<1x1x32x128xbf16> {
    // Chained unsqueezes are not converted optimaly. Ideally, we'd just use one operations, but 
    // this is an edge case. 
    // TODO: look into folding these into one unsqueeze in the future. 
    
    // First unsqueeze to 3D at dim 0
    %0 = ttir.empty() : tensor<1x32x128xbf16>
    %1 = "ttir.unsqueeze"(%arg0, %0) <{dim = 0 : si32}> : (tensor<32x128xbf16>, tensor<1x32x128xbf16>) -> tensor<1x32x128xbf16>
    
    // Then unsqueeze to 4D at dim 0
    %2 = ttir.empty() : tensor<1x1x32x128xbf16>
    %3 = "ttir.unsqueeze"(%1, %2) <{dim = 0 : si32}> : (tensor<1x32x128xbf16>, tensor<1x1x32x128xbf16>) -> tensor<1x1x32x128xbf16>
    
    // CHECK: %[[RESHAPE:.*]] = "ttnn.reshape"
    // CHECK-SAME: (tensor<32x128xbf16, {{.*}}>) -> tensor<1x32x128xbf16, {{.*}}>
    // CHECK: %[[UNSQUEEZE:.*]] = "ttnn.unsqueeze_to_4D"
    // CHECK-SAME: (tensor<1x32x128xbf16, {{.*}}>) -> tensor<1x1x32x128xbf16, {{.*}}>
    return %3 : tensor<1x1x32x128xbf16>
  }

  // CHECK-LABEL: func.func @unsqueeze_3d_to_4d_conversion
  func.func @unsqueeze_3d_to_4d_conversion(%arg0: tensor<8x32x128xbf16>) -> tensor<1x8x32x128xbf16> {
    %0 = ttir.empty() : tensor<1x8x32x128xbf16>
    %1 = "ttir.unsqueeze"(%arg0, %0) <{dim = 0 : si32}> : (tensor<8x32x128xbf16>, tensor<1x8x32x128xbf16>) -> tensor<1x8x32x128xbf16>
    
    // CHECK: %[[C:.*]] = "ttnn.unsqueeze_to_4D"
    // CHECK-SAME: (tensor<8x32x128xbf16, {{.*}}>) -> tensor<1x8x32x128xbf16, {{.*}}>
    return %1 : tensor<1x8x32x128xbf16>
  }

  // CHECK-LABEL: func.func @unsqueeze_not_to_4d_uses_reshape
  func.func @unsqueeze_not_to_4d_uses_reshape(%arg0: tensor<32x128xbf16>) -> tensor<1x32x128xbf16> {
    %0 = ttir.empty() : tensor<1x32x128xbf16>
    %1 = "ttir.unsqueeze"(%arg0, %0) <{dim = 0 : si32}> : (tensor<32x128xbf16>, tensor<1x32x128xbf16>) -> tensor<1x32x128xbf16>
    
    // CHECK-NOT: ttnn.unsqueeze_to_4D
    // CHECK: ttnn.reshape
    return %1 : tensor<1x32x128xbf16>
  }

  // CHECK-LABEL: func.func @unsqueeze_middle_dim_uses_reshape
  func.func @unsqueeze_middle_dim_uses_reshape(%arg0: tensor<8x32x128xbf16>) -> tensor<8x1x32x128xbf16> {
    %0 = ttir.empty() : tensor<8x1x32x128xbf16>
    %1 = "ttir.unsqueeze"(%arg0, %0) <{dim = 1 : si32}> : (tensor<8x32x128xbf16>, tensor<8x1x32x128xbf16>) -> tensor<8x1x32x128xbf16>
    
    // CHECK-NOT: ttnn.unsqueeze_to_4D
    // CHECK: ttnn.reshape
    return %1 : tensor<8x1x32x128xbf16>
  }
}