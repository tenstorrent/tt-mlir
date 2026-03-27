// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-greedy-optimizer=true" -o %t %s
// RUN: FileCheck %s --input-file=%t
//
// Test: sharded-to-sharded reshard candidates are explored.
// Matmul followed by elementwise -- matmul may produce a different shard spec
// than what the elementwise prefers. Verify the pipeline doesn't crash and
// produces valid layouts.

module attributes {} {
  func.func @forward(%arg0: tensor<1x1x128x256xbf16>, %arg1: tensor<1x1x256x128xbf16>,
                     %arg2: tensor<1x1x128x128xbf16>) -> tensor<1x1x128x128xbf16> {
    // CHECK: "ttnn.matmul"{{.*}} -> tensor<1x1x128x128xbf16, #{{.*}}>
    %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<1x1x128x256xbf16>, tensor<1x1x256x128xbf16>) -> tensor<1x1x128x128xbf16>
    // CHECK: "ttnn.add"{{.*}} -> tensor<1x1x128x128xbf16, #{{.*}}>
    %1 = "ttir.add"(%0, %arg2) : (tensor<1x1x128x128xbf16>, tensor<1x1x128x128xbf16>) -> tensor<1x1x128x128xbf16>
    return %1 : tensor<1x1x128x128xbf16>
  }
}
