// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-greedy-optimizer=true optimization-level=2" --mlir-print-local-scope -o %t %s
// RUN: FileCheck %s --input-file=%t
//
// Test: simple elementwise chain gets layout annotations from greedy optimizer.
// The optimizer should assign L1 layouts to all ops. A to_memory_config op
// is inserted before func.return to spill the result to DRAM.

module attributes {} {
  func.func @forward(%arg0: tensor<64x128xbf16>, %arg1: tensor<64x128xbf16>) -> tensor<64x128xbf16> {
    // CHECK: "ttnn.add"{{.*}} -> tensor<64x128xbf16, #ttnn.ttnn_layout<{{.*}}#ttnn.buffer_type<l1>{{.*}}>>
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    // CHECK: "ttnn.relu"{{.*}} -> tensor<64x128xbf16, #ttnn.ttnn_layout<{{.*}}#ttnn.buffer_type<l1>{{.*}}>>
    %1 = "ttir.relu"(%0) : (tensor<64x128xbf16>) -> tensor<64x128xbf16>
    // CHECK: "ttnn.multiply"{{.*}} -> tensor<64x128xbf16, #ttnn.ttnn_layout<{{.*}}#ttnn.buffer_type<l1>{{.*}}>>
    %2 = "ttir.multiply"(%1, %arg1) : (tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    // CHECK: "ttnn.to_memory_config"{{.*}} -> tensor<64x128xbf16, #ttnn.ttnn_layout<{{.*}}#ttnn.buffer_type<dram>{{.*}}>>
    // CHECK: return {{.*}} tensor<64x128xbf16, #ttnn.ttnn_layout<{{.*}}#ttnn.buffer_type<dram>{{.*}}>>
    return %2 : tensor<64x128xbf16>
  }
}
