// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-greedy-optimizer=true" -o %t %s
// RUN: FileCheck %s --input-file=%t
//
// Test: beam search resolves fork points without crashing.
// The add0 result is consumed by both relu and the second add.

module attributes {} {
  func.func @forward(%arg0: tensor<64x128xbf16>, %arg1: tensor<64x128xbf16>,
                     %arg2: tensor<64x128xbf16>) -> tensor<64x128xbf16> {
    // CHECK: %[[ADD0:.*]] = "ttnn.add"
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    // CHECK: %[[RELU:.*]] = "ttnn.relu"
    %1 = "ttir.relu"(%0) : (tensor<64x128xbf16>) -> tensor<64x128xbf16>
    // CHECK: "ttnn.multiply"
    %2 = "ttir.multiply"(%1, %arg2) : (tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    // Fork: %0 is used by relu above AND this add.
    // CHECK: "ttnn.add"
    %3 = "ttir.add"(%2, %0) : (tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    return %3 : tensor<64x128xbf16>
  }
}
