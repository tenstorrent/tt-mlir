// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-greedy-optimizer=true" -o %t %s
// RUN: FileCheck %s --input-file=%t
//
// Test: fork pattern (residual connection). A tensor is consumed by multiple ops.
// Greedy optimizer should handle this without crashing.

// CHECK-DAG: #[[LAYOUT:.*]] = #ttnn.ttnn_layout<{{.*}}>
module attributes {} {
  func.func @forward(%arg0: tensor<64x128xbf16>, %arg1: tensor<64x128xbf16>) -> tensor<64x128xbf16> {
    // This add result is used by both relu and the final add (fork).
    // CHECK: %[[ADD0:.*]] = "ttnn.add"
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    // CHECK: %[[RELU:.*]] = "ttnn.relu"
    %1 = "ttir.relu"(%0) : (tensor<64x128xbf16>) -> tensor<64x128xbf16>
    // Residual connection: add the fork tensor with its transformed version.
    // CHECK: "ttnn.add"(%[[RELU]], %[[ADD0]])
    %2 = "ttir.add"(%1, %0) : (tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    return %2 : tensor<64x128xbf16>
  }
}
