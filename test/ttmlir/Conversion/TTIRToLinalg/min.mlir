// RUN: ttmlir-opt --convert-ttir-to-linalg -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  func.func @min_test(%arg0: tensor<1x1x49x2048xf32>) -> tensor<1x1x1x2048xf32> {
    // CHECK: tosa.reduce_min
    %1 = "ttir.min"(%arg0) <{dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x1x49x2048xf32>) -> tensor<1x1x1x2048xf32>
    return %1 : tensor<1x1x1x2048xf32>
  }
  func.func @min_test_neg(%arg0: tensor<1x1x49x2048xf32>) -> tensor<1x1x1x2048xf32> {
    // CHECK: tosa.reduce_min
    %1 = "ttir.min"(%arg0) <{dim_arg = [-2 : i32], keep_dim = true}> : (tensor<1x1x49x2048xf32>) -> tensor<1x1x1x2048xf32>
    return %1 : tensor<1x1x1x2048xf32>
  }
  func.func @min_test_array(%arg0: tensor<1x10x10x2048xf32>) -> tensor<1x1x1x2048xf32> {
    // CHECK: tosa.reduce_min
    // CHECK: tosa.reduce_min
    %1 = "ttir.min"(%arg0) <{dim_arg = [1 : i32, -2: i32], keep_dim = true}> : (tensor<1x10x10x2048xf32>) -> tensor<1x1x1x2048xf32>
    return %1 : tensor<1x1x1x2048xf32>
  }
  func.func @min_test_no_keep(%arg0: tensor<1x1x49x2048xf32>) -> tensor<1x1x2048xf32> {
    // CHECK: tosa.reduce_min
    // CHECK: tosa.reshape
    %1 = "ttir.min"(%arg0) <{dim_arg = [-2 : i32], keep_dim = false}> : (tensor<1x1x49x2048xf32>) -> tensor<1x1x2048xf32>
    return %1 : tensor<1x1x2048xf32>
  }
  func.func @min_test_no_keep_array(%arg0: tensor<1x10x10x2048xf32>) -> tensor<1x2048xf32> {
    // CHECK: tosa.reduce_min
    // CHECK: tosa.reduce_min
    // CHECK: tosa.reshape
    %1 = "ttir.min"(%arg0) <{dim_arg = [1 : i32, -2 : i32], keep_dim = false}> : (tensor<1x10x10x2048xf32>) -> tensor<1x2048xf32>
    return %1 : tensor<1x2048xf32>
  }
}
