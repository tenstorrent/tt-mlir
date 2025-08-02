// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-bfp8-type=true" %s | FileCheck %s

module  {
  // CHECK-LABEL: @forward(
  // CHECK-SAME: tensor<32x32x!ttcore.bfp8_b
  // CHECK-SAME: tensor<32x32x!ttcore.bfp8_b
  // CHECK-SAME: -> tensor<32x32x!ttcore.bfp8_b
  func.func @forward(%arg0 : tensor<32x32xbf16>, %arg1 : tensor<32x32xbf16>) ->tensor<32x32xbf16> {
    %0 = ttir.empty() : tensor<32x32xbf16>
    // CHECK: "ttnn.add"
    // CHECK-SAME: tensor<32x32x!ttcore.bfp8_b
    // CHECK-SAME: tensor<32x32x!ttcore.bfp8_b
    // CHECK-SAME: -> tensor<32x32x!ttcore.bfp8_b
    %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %1 : tensor<32x32xbf16>
  }
}
