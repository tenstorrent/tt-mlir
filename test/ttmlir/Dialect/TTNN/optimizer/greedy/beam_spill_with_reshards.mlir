// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-greedy-optimizer=true tensor-l1-usage-cap=0.01" -o %t %s
// RUN: FileCheck %s --input-file=%t
//
// Test: beam search + reshards under tight L1 budget.
// Verify L1 spill management works correctly with reshard ops.

module attributes {} {
  func.func @forward(%arg0: tensor<512x512xbf16>, %arg1: tensor<512x512xbf16>,
                     %arg2: tensor<512x512xbf16>) -> tensor<512x512xbf16> {
    // CHECK: "ttnn.add"{{.*}} -> tensor<512x512xbf16, #{{.*}}>
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<512x512xbf16>, tensor<512x512xbf16>) -> tensor<512x512xbf16>
    // CHECK: "ttnn.multiply"{{.*}} -> tensor<512x512xbf16, #{{.*}}>
    %1 = "ttir.multiply"(%arg0, %arg2) : (tensor<512x512xbf16>, tensor<512x512xbf16>) -> tensor<512x512xbf16>
    // CHECK: "ttnn.add"{{.*}} -> tensor<512x512xbf16, #{{.*}}>
    %2 = "ttir.add"(%0, %1) : (tensor<512x512xbf16>, tensor<512x512xbf16>) -> tensor<512x512xbf16>
    return %2 : tensor<512x512xbf16>
  }
}
