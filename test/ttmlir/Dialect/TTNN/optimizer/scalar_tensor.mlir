// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=2" -o %t %s
// RUN: FileCheck %s --input-file=%t
module attributes {} {
  // CHECK-LABEL: func.func @scalar_add
  func.func @scalar_add(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %1 = "ttir.add"(%0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %2 = "ttir.add"(%1, %arg0) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    %3 = "ttir.add"(%2, %1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
    return %3 : tensor<f32>
  }
}
