// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module {
  func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32>
    %1 = call @mul_by_two(%0) : (tensor<4xf32>) -> tensor<4xf32>
    %2 = call @add_three(%1) : (tensor<4xf32>) -> tensor<4xf32>
    // CHECK: = "ttir.constant"
    // CHECK: = "ttir.constant"
    // CHECK: = ttir.empty
    // CHECK: = "ttir.add"
    // CHECK: = ttir.empty
    // CHECK: = "ttir.broadcast"
    // CHECK: = ttir.empty
    // CHECK: = "ttir.multiply"
    // CHECK: = ttir.empty
    // CHECK: = "ttir.broadcast"
    // CHECK: = ttir.empty
    // CHECK: = "ttir.add"
    return %2 : tensor<4xf32>
  }

  func.func @mul_by_two(%x: tensor<4xf32>) -> tensor<4xf32> {
    %c2 = stablehlo.constant dense<2.0> : tensor<f32>
    %c2_broadcast = stablehlo.broadcast_in_dim %c2, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %result = stablehlo.multiply %x, %c2_broadcast : tensor<4xf32>
    // CHECK: = "ttir.constant"
    // CHECK: = ttir.empty
    // CHECK: = "ttir.broadcast"
    // CHECK: = ttir.empty
    // CHECK: = "ttir.multiply"
    return %result : tensor<4xf32>
  }

  func.func @add_three(%x: tensor<4xf32>) -> tensor<4xf32> {
    %c3 = stablehlo.constant dense<3.0> : tensor<f32>
    %c3_broadcast = stablehlo.broadcast_in_dim %c3, dims = [] : (tensor<f32>) -> tensor<4xf32>
    %result = stablehlo.add %x, %c3_broadcast : tensor<4xf32>
    // CHECK: = "ttir.constant"
    // CHECK: = ttir.empty
    // CHECK: = "ttir.broadcast"
    // CHECK: = ttir.empty
    // CHECK: = "ttir.add"
    return %result : tensor<4xf32>
  }
}
