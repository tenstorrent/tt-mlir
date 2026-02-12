// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-greedy-optimizer=true tensor-l1-usage-cap=0.01" -o %t %s
// RUN: FileCheck %s --input-file=%t
//
// Test: With an extremely low L1 usage cap (1%), Belady's algorithm should
// enforce L1 budget. Verify the pass runs without crashing and produces
// valid TTNN IR with layout annotations.

module attributes {} {
  func.func @forward(%arg0: tensor<512x512xbf16>, %arg1: tensor<512x512xbf16>,
                     %arg2: tensor<512x512xbf16>) -> tensor<512x512xbf16> {
    // CHECK: "ttnn.add"{{.*}} -> tensor<512x512xbf16, #{{.*}}>
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<512x512xbf16>, tensor<512x512xbf16>) -> tensor<512x512xbf16>
    // CHECK: "ttnn.multiply"{{.*}} -> tensor<512x512xbf16, #{{.*}}>
    %1 = "ttir.multiply"(%arg0, %arg2) : (tensor<512x512xbf16>, tensor<512x512xbf16>) -> tensor<512x512xbf16>
    // CHECK: "ttnn.add"{{.*}} -> tensor<512x512xbf16, #{{.*}}>
    %2 = "ttir.add"(%0, %1) : (tensor<512x512xbf16>, tensor<512x512xbf16>) -> tensor<512x512xbf16>
    // CHECK: "ttnn.relu"{{.*}} -> tensor<512x512xbf16, #{{.*}}>
    %3 = "ttir.relu"(%2) : (tensor<512x512xbf16>) -> tensor<512x512xbf16>
    // Verify no L1 usage attributes remain (cleaned up).
    // CHECK-NOT: ttnn.output_l1_usage
    return %3 : tensor<512x512xbf16>
  }
}
