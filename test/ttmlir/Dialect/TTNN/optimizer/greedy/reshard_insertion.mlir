// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-greedy-optimizer=true" -o %t %s
// RUN: FileCheck %s --input-file=%t
//
// Test: matmul followed by an elementwise op. Both should get layouts from
// the greedy optimizer and compile successfully.

module attributes {} {
  func.func @forward(%arg0: tensor<32x128xbf16>, %arg1: tensor<128x64xbf16>,
                     %arg2: tensor<32x64xbf16>) -> tensor<32x64xbf16> {
    // CHECK: "ttnn.matmul"{{.*}} -> tensor<32x64xbf16, #{{.*}}>
    %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<32x128xbf16>, tensor<128x64xbf16>) -> tensor<32x64xbf16>
    // CHECK: "ttnn.add"{{.*}} -> tensor<32x64xbf16, #{{.*}}>
    %1 = "ttir.add"(%0, %arg2) : (tensor<32x64xbf16>, tensor<32x64xbf16>) -> tensor<32x64xbf16>
    return %1 : tensor<32x64xbf16>
  }
}
