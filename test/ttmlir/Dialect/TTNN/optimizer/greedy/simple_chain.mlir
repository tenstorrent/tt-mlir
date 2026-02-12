// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-greedy-optimizer=true" -o %t %s
// RUN: FileCheck %s --input-file=%t
//
// Test: simple elementwise chain gets layout annotations from greedy optimizer.
// The optimizer should assign L1 layouts to intermediate ops. The last op
// (whose output feeds func.return) stays in DRAM since function outputs
// must be in DRAM for the caller.

// CHECK-DAG: #[[L1_LAYOUT:.*]] = #ttnn.ttnn_layout<{{.*}}#l1{{.*}}>
// CHECK-DAG: #[[DRAM_LAYOUT:.*]] = #ttnn.ttnn_layout<{{.*}}#dram{{.*}}>
module attributes {} {
  func.func @forward(%arg0: tensor<64x128xbf16>, %arg1: tensor<64x128xbf16>) -> tensor<64x128xbf16> {
    // CHECK: "ttnn.add"{{.*}} -> tensor<64x128xbf16, #[[L1_LAYOUT]]>
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    // CHECK: "ttnn.relu"{{.*}} -> tensor<64x128xbf16, #[[L1_LAYOUT]]>
    %1 = "ttir.relu"(%0) : (tensor<64x128xbf16>) -> tensor<64x128xbf16>
    // CHECK: "ttnn.multiply"{{.*}} -> tensor<64x128xbf16, #[[DRAM_LAYOUT]]>
    %2 = "ttir.multiply"(%1, %arg1) : (tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    // CHECK: return %{{.*}} : tensor<64x128xbf16, #[[DRAM_LAYOUT]]>
    return %2 : tensor<64x128xbf16>
  }
}
