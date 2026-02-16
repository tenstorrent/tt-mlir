// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-greedy-optimizer=true" -o %t %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t
// RUN: FileCheck %s --input-file=%t
//
// Test: end-to-end with beam search produces valid flatbuffer.
// Fork pattern with larger tensor shapes.

module attributes {} {
  func.func @forward(%arg0: tensor<1x1x256x256xbf16>, %arg1: tensor<1x1x256x256xbf16>,
                     %arg2: tensor<1x1x256x256xbf16>) -> tensor<1x1x256x256xbf16> {
    // CHECK: "ttnn.add"
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<1x1x256x256xbf16>, tensor<1x1x256x256xbf16>) -> tensor<1x1x256x256xbf16>
    // CHECK: "ttnn.relu"
    %1 = "ttir.relu"(%0) : (tensor<1x1x256x256xbf16>) -> tensor<1x1x256x256xbf16>
    // CHECK: "ttnn.multiply"
    %2 = "ttir.multiply"(%1, %arg2) : (tensor<1x1x256x256xbf16>, tensor<1x1x256x256xbf16>) -> tensor<1x1x256x256xbf16>
    // Fork: %0 used here too.
    // CHECK: "ttnn.add"
    %3 = "ttir.add"(%2, %0) : (tensor<1x1x256x256xbf16>, tensor<1x1x256x256xbf16>) -> tensor<1x1x256x256xbf16>
    return %3 : tensor<1x1x256x256xbf16>
  }
}
