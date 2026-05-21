// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=1" -o %t %s
// RUN: FileCheck %s --input-file=%t
module attributes {} {
  func.func @forward(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %1 = "ttir.multiply"(%arg0, %arg1) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    // CHECK: "ttnn.multiply"
    return %1 : tensor<64x128xf32>
  }
  // CHECK-LABEL: func.func @scalar_add
  func.func @scalar_add(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    return %0 : tensor<f32>
  }
}
