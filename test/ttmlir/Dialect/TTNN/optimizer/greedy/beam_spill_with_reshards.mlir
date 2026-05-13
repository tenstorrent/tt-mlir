// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-greedy-optimizer=true tensor-l1-usage-cap=0.01" -o %t %s
// RUN: FileCheck %s --input-file=%t
//
// Test: L1 spill management under tight L1 budget (1% cap).
// Verify ops are demoted to DRAM and the pass produces valid TTNN IR.

// Verify outputs use DRAM buffer type under 1% cap.
// CHECK: #dram = #ttnn.buffer_type<dram>

module attributes {} {
  func.func @forward(%arg0: tensor<512x512xbf16>, %arg1: tensor<512x512xbf16>,
                     %arg2: tensor<512x512xbf16>) -> tensor<512x512xbf16> {
    // CHECK: "ttnn.add"{{.*}} -> tensor<512x512xbf16, #ttnn_layout>
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<512x512xbf16>, tensor<512x512xbf16>) -> tensor<512x512xbf16>
    // CHECK: "ttnn.multiply"{{.*}} -> tensor<512x512xbf16, #ttnn_layout>
    %1 = "ttir.multiply"(%arg0, %arg2) : (tensor<512x512xbf16>, tensor<512x512xbf16>) -> tensor<512x512xbf16>
    // CHECK: "ttnn.add"{{.*}} -> tensor<512x512xbf16, #ttnn_layout>
    %2 = "ttir.add"(%0, %1) : (tensor<512x512xbf16>, tensor<512x512xbf16>) -> tensor<512x512xbf16>
    // CHECK-NOT: ttnn.output_l1_usage
    return %2 : tensor<512x512xbf16>
  }
}
