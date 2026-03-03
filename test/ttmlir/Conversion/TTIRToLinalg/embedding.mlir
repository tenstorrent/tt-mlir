// RUN: ttmlir-opt --convert-ttir-to-linalg -o %t %s
// RUN: FileCheck %s --input-file=%t

module attributes {} {
  // CHECK-LABEL: func.func @test_embedding_2d_input
  func.func @test_embedding_2d_input(%arg0: tensor<2x3xi32>, %arg1: tensor<10x4xf32>) -> tensor<2x3x4xf32> {
    // CHECK: tensor.empty()
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%arg0 : tensor<2x3xi32>)
    // CHECK: arith.index_cast
    // CHECK: tensor.extract %arg1
    // CHECK: linalg.yield
    %0 = "ttir.embedding"(%arg0, %arg1) : (tensor<2x3xi32>, tensor<10x4xf32>) -> tensor<2x3x4xf32>
    // CHECK: return %{{[0-9]+}} : tensor<2x3x4xf32>
    return %0 : tensor<2x3x4xf32>
  }

  // CHECK-LABEL: func.func @test_embedding_1d_input
  func.func @test_embedding_1d_input(%arg0: tensor<4xi32>, %arg1: tensor<10x8xf32>) -> tensor<4x8xf32> {
    // CHECK: tensor.empty()
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%arg0 : tensor<4xi32>)
    // CHECK: arith.index_cast
    // CHECK: tensor.extract %arg1
    // CHECK: linalg.yield
    %0 = "ttir.embedding"(%arg0, %arg1) : (tensor<4xi32>, tensor<10x8xf32>) -> tensor<4x8xf32>
    // CHECK: return %{{[0-9]+}} : tensor<4x8xf32>
    return %0 : tensor<4x8xf32>
  }
}
