// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-greedy-optimizer=true" -o %t %s
// RUN: FileCheck %s --input-file=%t
//
// Test: small model with comfortable L1 headroom. The spill management pass
// should not insert any spill-to-DRAM ops. All operations should compile
// without forced DRAM spills.

module attributes {} {
  func.func @forward(%arg0: tensor<64x128xbf16>, %arg1: tensor<64x128xbf16>) -> tensor<64x128xbf16> {
    // CHECK: "ttnn.add"{{.*}} -> tensor<64x128xbf16, #{{.*}}>
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    // CHECK: "ttnn.relu"{{.*}} -> tensor<64x128xbf16, #{{.*}}>
    %1 = "ttir.relu"(%0) : (tensor<64x128xbf16>) -> tensor<64x128xbf16>
    // Verify no ttnn.output_l1_usage attributes remain (cleaned up by spill pass).
    // CHECK-NOT: ttnn.output_l1_usage
    return %1 : tensor<64x128xbf16>
  }
}
