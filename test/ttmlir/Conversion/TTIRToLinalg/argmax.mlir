// RUN: ttmlir-opt --convert-ttir-to-linalg -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // CHECK-LABEL: func.func @argmax_dim0
  func.func @argmax_dim0(%arg0: tensor<4x8xf32>) -> tensor<8xi32> {
    // CHECK: linalg.fill
    // CHECK: linalg.fill
    // CHECK: linalg.generic
    // CHECK: linalg.index 0
    // CHECK: arith.cmpf ogt
    // CHECK: arith.select
    // CHECK: arith.select
    // CHECK: linalg.yield
    %0 = "ttir.argmax"(%arg0) <{dim_arg = [0 : i32], keep_dim = false}> : (tensor<4x8xf32>) -> tensor<8xi32>
    return %0 : tensor<8xi32>
  }

  // CHECK-LABEL: func.func @argmax_dim1
  func.func @argmax_dim1(%arg0: tensor<4x8xf32>) -> tensor<4xi32> {
    // CHECK: linalg.fill
    // CHECK: linalg.fill
    // CHECK: linalg.generic
    // CHECK: linalg.index 1
    // CHECK: arith.cmpf ogt
    // CHECK: arith.select
    // CHECK: arith.select
    // CHECK: linalg.yield
    %0 = "ttir.argmax"(%arg0) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<4x8xf32>) -> tensor<4xi32>
    return %0 : tensor<4xi32>
  }

  // CHECK-LABEL: func.func @argmax_dim0_keep_dim
  func.func @argmax_dim0_keep_dim(%arg0: tensor<4x8xf32>) -> tensor<1x8xi32> {
    // CHECK: linalg.generic
    // CHECK: arith.cmpf ogt
    // CHECK: arith.select
    // CHECK: arith.select
    // CHECK: linalg.yield
    // CHECK: tosa.const_shape
    // CHECK: tosa.reshape
    %0 = "ttir.argmax"(%arg0) <{dim_arg = [0 : i32], keep_dim = true}> : (tensor<4x8xf32>) -> tensor<1x8xi32>
    return %0 : tensor<1x8xi32>
  }

  // CHECK-LABEL: func.func @argmax_dim1_keep_dim
  func.func @argmax_dim1_keep_dim(%arg0: tensor<4x8xf32>) -> tensor<4x1xi32> {
    // CHECK: linalg.generic
    // CHECK: arith.cmpf ogt
    // CHECK: arith.select
    // CHECK: arith.select
    // CHECK: linalg.yield
    // CHECK: tosa.const_shape
    // CHECK: tosa.reshape
    %0 = "ttir.argmax"(%arg0) <{dim_arg = [1 : i32], keep_dim = true}> : (tensor<4x8xf32>) -> tensor<4x1xi32>
    return %0 : tensor<4x1xi32>
  }

  // CHECK-LABEL: func.func @argmax_all_dims
  func.func @argmax_all_dims(%arg0: tensor<4x8xf32>) -> tensor<i32> {
    // CHECK: linalg.generic
    // CHECK: linalg.index 0
    // CHECK: linalg.index 1
    // CHECK: arith.muli
    // CHECK: arith.addi
    // CHECK: arith.cmpf ogt
    // CHECK: arith.select
    // CHECK: arith.select
    // CHECK: linalg.yield
    %0 = "ttir.argmax"(%arg0) <{keep_dim = false}> : (tensor<4x8xf32>) -> tensor<i32>
    return %0 : tensor<i32>
  }
}
