// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-greedy-optimizer=true" -o %t %s
// RUN: FileCheck %s --input-file=%t
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t
//
// Test: end-to-end pipeline integration. Verify the greedy optimizer produces
// valid TTNN IR that can be translated to flatbuffer.

module attributes {} {
  func.func @forward(%arg0: tensor<64x128xbf16>, %arg1: tensor<64x128xbf16>,
                     %arg2: tensor<64x128xbf16>) -> tensor<64x128xbf16> {
    // CHECK: "ttnn.add"
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    // CHECK: "ttnn.relu"
    %1 = "ttir.relu"(%0) : (tensor<64x128xbf16>) -> tensor<64x128xbf16>
    // Residual connection (fork pattern).
    // CHECK: "ttnn.multiply"
    %2 = "ttir.multiply"(%1, %arg2) : (tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    // CHECK: "ttnn.add"
    %3 = "ttir.add"(%2, %0) : (tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    // CHECK: return
    return %3 : tensor<64x128xbf16>
  }
}
