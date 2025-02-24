// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
module {
  func.func @forward(%arg0: tensor<1x11x2048xf32>, %arg1: tensor<2048x128256xf32>) -> tensor<1x11x128256xf32> {
    %0 = tensor.empty() : tensor<1x11x128256xf32>
    // CHECK: "ttnn.matmul"
    %1 = "ttir.matmul"(%arg0, %arg1, %0) : (tensor<1x11x2048xf32>, tensor<2048x128256xf32>, tensor<1x11x128256xf32>) -> tensor<1x11x128256xf32>
    return %1 : tensor<1x11x128256xf32>
  }
}
