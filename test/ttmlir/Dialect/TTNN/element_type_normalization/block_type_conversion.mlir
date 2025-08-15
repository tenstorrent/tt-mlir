// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-bfp8-conversion=true" %s | FileCheck %s

module  {
  // CHECK-LABEL: @forward(
  // CHECK-SAME: tensor<32x32xbf16
  // CHECK-SAME: tensor<32x32xbf16
  // CHECK-SAME: -> tensor<32x32xbf16
  func.func @forward(%arg0 : tensor<32x32xbf16>, %arg1 : tensor<32x32xbf16>) ->tensor<32x32xbf16> {
    %0 = ttir.empty() : tensor<32x32xbf16>
    // CHECK: "ttnn.add"
    // CHECK-SAME: tensor<32x32x!ttcore.tile<32x32, bfp_bf8
    // CHECK-SAME: tensor<32x32x!ttcore.tile<32x32, bfp_bf8
    // CHECK-SAME: -> tensor<32x32x!ttcore.tile<32x32, bfp_bf8
    %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    // CHECK: return
    // CHECK-SAME: tensor<32x32xbf16
    return %1 : tensor<32x32xbf16>
  }
}
