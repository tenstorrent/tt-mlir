// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-greedy-optimizer=true" -o %t %s
// RUN: FileCheck %s --input-file=%t
//
// Test: reshape op gets a valid layout from the greedy optimizer.
// The optimizer should assign a layout (the specific buffer type depends
// on what the backend validates for the given shape).

module attributes {} {
  func.func @forward(%arg0: tensor<1x32x64xbf16>) -> tensor<1x2048xbf16> {
    // CHECK: "ttnn.reshape"{{.*}} -> tensor<1x2048xbf16, #{{.*}}>
    %0 = "ttir.reshape"(%arg0) <{shape = [1 : i32, 2048 : i32]}> : (tensor<1x32x64xbf16>) -> tensor<1x2048xbf16>
    return %0 : tensor<1x2048xbf16>
  }
}
