// RUN: ttmlir-opt --convert-ttir-to-linalg -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  func.func @mean_test(%arg0: tensor<1x1x49x2048xbf16>) -> tensor<1x1x1x2048xbf16> {
    %0 = ttir.empty() : tensor<1x1x1x2048xbf16>
    // CHECK: tosa.reduce_sum
    // CHECK: tosa.const
    // CHECK: linalg.div
    %1 = "ttir.mean"(%arg0, %0) <{dim_arg = [2 : i32], keep_dim = true}> : (tensor<1x1x49x2048xbf16>, tensor<1x1x1x2048xbf16>) -> tensor<1x1x1x2048xbf16>
    return %1 : tensor<1x1x1x2048xbf16>
  }
  func.func @mean_test_neg(%arg0: tensor<1x1x49x2048xbf16>) -> tensor<1x1x1x2048xbf16> {
    %0 = ttir.empty() : tensor<1x1x1x2048xbf16>
    // CHECK: tosa.reduce_sum
    // CHECK: tosa.const
    // CHECK: linalg.div
    %1 = "ttir.mean"(%arg0, %0) <{dim_arg = [-2 : i32], keep_dim = true}> : (tensor<1x1x49x2048xbf16>, tensor<1x1x1x2048xbf16>) -> tensor<1x1x1x2048xbf16>
    return %1 : tensor<1x1x1x2048xbf16>
  }
  func.func @mean_test_array(%arg0: tensor<1x10x10x2048xbf16>) -> tensor<1x1x1x2048xbf16> {
    %0 = ttir.empty() : tensor<1x1x1x2048xbf16>
    // CHECK: tosa.reduce_sum
    // CHECK: tosa.reduce_sum
    // CHECK: tosa.const
    // CHECK: linalg.div
    %1 = "ttir.mean"(%arg0, %0) <{dim_arg = [1 : i32, -2: i32], keep_dim = true}> : (tensor<1x10x10x2048xbf16>, tensor<1x1x1x2048xbf16>) -> tensor<1x1x1x2048xbf16>
    return %1 : tensor<1x1x1x2048xbf16>
  }
  func.func @mean_test_no_keep(%arg0: tensor<1x1x49x2048xbf16>) -> tensor<1x1x2048xbf16> {
    %0 = ttir.empty() : tensor<1x1x2048xbf16>
    // CHECK: tosa.reduce_sum
    // CHECK: tosa.reshape
    // CHECK: tosa.const
    // CHECK: linalg.div
    %1 = "ttir.mean"(%arg0, %0) <{dim_arg = [-2 : i32], keep_dim = false}> : (tensor<1x1x49x2048xbf16>, tensor<1x1x2048xbf16>) -> tensor<1x1x2048xbf16>
    return %1 : tensor<1x1x2048xbf16>
  }
    func.func @mean_test_no_keep_array(%arg0: tensor<1x10x10x2048xbf16>) -> tensor<1x2048xbf16> {
    %0 = ttir.empty() : tensor<1x2048xbf16>
    // CHECK: tosa.reduce_sum
    // CHECK: tosa.reduce_sum
    // CHECK: tosa.reshape
    // CHECK: tosa.const
    // CHECK: linalg.div
    %1 = "ttir.mean"(%arg0, %0) <{dim_arg = [1 : i32, -2 : i32], keep_dim = false}> : (tensor<1x10x10x2048xbf16>, tensor<1x2048xbf16>) -> tensor<1x2048xbf16>
    return %1 : tensor<1x2048xbf16>
  }

}
