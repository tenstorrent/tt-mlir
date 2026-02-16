// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-greedy-optimizer=true" -o %t %s
// RUN: FileCheck %s --input-file=%t
//
// Test: beam search (K=8) on a simple chain produces valid layouts.
// Should be equivalent to greedy for a linear chain (no forks).

// CHECK-DAG: #[[L1_LAYOUT:.*]] = #ttnn.ttnn_layout<{{.*}}#l1{{.*}}>
module attributes {} {
  func.func @forward(%arg0: tensor<64x128xbf16>, %arg1: tensor<64x128xbf16>) -> tensor<64x128xbf16> {
    // CHECK: "ttnn.add"{{.*}} -> tensor<64x128xbf16, #[[L1_LAYOUT]]>
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    // CHECK: "ttnn.relu"{{.*}} -> tensor<64x128xbf16, #[[L1_LAYOUT]]>
    %1 = "ttir.relu"(%0) : (tensor<64x128xbf16>) -> tensor<64x128xbf16>
    %2 = "ttir.multiply"(%1, %arg1) : (tensor<64x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    return %2 : tensor<64x128xbf16>
  }
}
