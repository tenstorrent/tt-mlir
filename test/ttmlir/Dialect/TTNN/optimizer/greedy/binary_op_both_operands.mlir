// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-greedy-optimizer=true" -o %t %s
// RUN: FileCheck %s --input-file=%t
//
// Test: binary op considers both operands (not just operand 0).
// Two independent producer chains feed into a single add.

module attributes {} {
  func.func @forward(%arg0: tensor<64x128xbf16>, %arg1: tensor<64x128xbf16>,
                     %arg2: tensor<64x128xbf16>, %arg3: tensor<64x128xbf16>) -> tensor<64x128xbf16> {
    // CHECK: "ttnn.add"
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    // CHECK: "ttnn.multiply"
    %1 = "ttir.multiply"(%arg2, %arg3) : (tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    // The final add consumes both operands -- greedy optimizer should handle both.
    // CHECK: "ttnn.add"{{.*}} -> tensor<64x128xbf16, #{{.*}}>
    %2 = "ttir.add"(%0, %1) : (tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    return %2 : tensor<64x128xbf16>
  }
}
