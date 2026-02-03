// RUN: ttmlir-opt --convert-ttir-to-linalg -o %t %s
// RUN: FileCheck %s --input-file=%t

module attributes {} {
  func.func @test_linear_2d(%arg0: tensor<10x32xf32>, %arg1: tensor<32x64xf32>) -> tensor<10x64xf32> {
    // CHECK: tosa.reshape %arg0
    // CHECK: tosa.reshape %arg1
    // CHECK: tosa.matmul
    // CHECK: tosa.reshape
    // CHECK-NOT: linalg.copy
    %1 = "ttir.linear"(%arg0, %arg1) : (tensor<10x32xf32>, tensor<32x64xf32>) -> tensor<10x64xf32>
    return %1 : tensor<10x64xf32>
  }

  func.func @test_linear_3d(%arg0: tensor<10x64x32xf32>, %arg1: tensor<32x128xf32>) -> tensor<10x64x128xf32> {
    // CHECK: tosa.reshape %arg1
    // CHECK: tosa.matmul
    // CHECK-NOT: linalg.copy
    %1 = "ttir.linear"(%arg0, %arg1) : (tensor<10x64x32xf32>, tensor<32x128xf32>) -> tensor<10x64x128xf32>
    return %1 : tensor<10x64x128xf32>
  }

  func.func @test_linear_with_bias(%arg0: tensor<10x64x32xf32>, %arg1: tensor<32x128xf32>, %arg2: tensor<128xf32>) -> tensor<10x64x128xf32> {
    // CHECK: tosa.matmul
    // CHECK: tosa.add
    // CHECK-NOT: linalg.copy
    %1 = "ttir.linear"(%arg0, %arg1, %arg2) : (tensor<10x64x32xf32>, tensor<32x128xf32>, tensor<128xf32>) -> tensor<10x64x128xf32>
    return %1 : tensor<10x64x128xf32>
  }
}
