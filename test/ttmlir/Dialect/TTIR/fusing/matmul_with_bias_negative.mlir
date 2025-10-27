// RUN: ttmlir-opt -ttir-to-ttir-decomposition -ttir-implicit-broadcast-fold -ttir-fusing -o %t %s
// RUN: FileCheck %s --input-file=%t

// ===----------------------------------------------------------------------===
// NEGATIVE CASES: Operations that should NOT be fused into linear op with bias
// ===----------------------------------------------------------------------===

// Dot general op has more than 1 user
module {
  func.func @dot_general_with_bias_1(%arg0: tensor<68x1024xf32>, %arg1: tensor<1024x1024xf32>, %arg2: tensor<68x1024xf32>) -> tensor<68x1024xf32> {
    // CHECK: func.func @dot_general_with_bias_1
    // CHECK-NOT: "ttir.linear"
    // CHECK: "ttir.matmul"
    // CHECK: "ttir.add"
    // CHECK: "ttir.subtract"
    %0 = ttir.empty() : tensor<68x1024xf32>
    %1 = "ttir.dot_general"(%arg0, %arg1) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<68x1024xf32>, tensor<1024x1024xf32>) -> (tensor<68x1024xf32>)
    %2 = ttir.empty() : tensor<68x1024xf32>
    %3 = "ttir.add"(%1, %arg2, %2) : (tensor<68x1024xf32>, tensor<68x1024xf32>, tensor<68x1024xf32>) -> tensor<68x1024xf32>
    %4 = ttir.empty() : tensor<2x34x16x64xf32>
    %5 = "ttir.reshape"(%3, %4)<{shape = [2 : i32, 34 : i32, 16 : i32, 64 : i32]}> : (tensor<68x1024xf32>, tensor<2x34x16x64xf32>) -> tensor<2x34x16x64xf32>
    %6 = ttir.empty() : tensor<68x1024xf32>
    %7 = "ttir.subtract"(%1, %3, %6) : (tensor<68x1024xf32>, tensor<68x1024xf32>, tensor<68x1024xf32>) -> tensor<68x1024xf32>
    return %7 : tensor<68x1024xf32>
  }
}

// Dot general and add operations are not feeding into one another
module {
  func.func @dot_general_with_bias_2(%arg0: tensor<68x1024xf32>, %arg1: tensor<1024x1024xf32>, %arg2: tensor<68x1024xf32>) -> tensor<68x1024xf32> {
    // CHECK: func.func @dot_general_with_bias_2
    // CHECK-NOT: "ttir.linear"
    // CHECK: "ttir.matmul"
    // CHECK: "ttir.add"
    // CHECK: "ttir.subtract"
    %0 = ttir.empty() : tensor<68x1024xf32>
    %1 = "ttir.dot_general"(%arg0, %arg1) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<68x1024xf32>, tensor<1024x1024xf32>) -> (tensor<68x1024xf32>)
    %2 = ttir.empty() : tensor<68x1024xf32>
    %3 = "ttir.add"(%arg0, %arg2, %2) : (tensor<68x1024xf32>, tensor<68x1024xf32>, tensor<68x1024xf32>) -> tensor<68x1024xf32>
    %6 = ttir.empty() : tensor<68x1024xf32>
    %7 = "ttir.subtract"(%1, %3, %6) : (tensor<68x1024xf32>, tensor<68x1024xf32>, tensor<68x1024xf32>) -> tensor<68x1024xf32>
    return %7 : tensor<68x1024xf32>
  }
}

// Bias and matmul are nor broadcast compatible, even though add follows matmul.
module {
  func.func @dot_general_with_bias_3(%arg0: tensor<68x1024xf32>, %arg1: tensor<1024x1024xf32>, %bias: tensor<2x2x34x1024xf32>) -> tensor<2x2x34x1024xf32> {
    // CHECK: func.func @dot_general_with_bias_3
    // CHECK-NOT: "ttir.linear"
    // CHECK: "ttir.matmul"
    // CHECK: "ttir.add"
    %0 = ttir.empty() : tensor<68x1024xf32>
    %1 = "ttir.dot_general"(%arg0, %arg1) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<68x1024xf32>, tensor<1024x1024xf32>) -> tensor<68x1024xf32>
    %2 = ttir.empty() : tensor<2x34x1024xf32>
    %3 = "ttir.reshape"(%1, %2) <{shape = [2 : i32, 34 : i32, 1024 : i32]}> : (tensor<68x1024xf32>, tensor<2x34x1024xf32>) -> tensor<2x34x1024xf32>
    %4 = ttir.empty() : tensor<2x2x34x1024xf32>
    %5 = "ttir.add"(%3, %bias, %4) : (tensor<2x34x1024xf32>, tensor<2x2x34x1024xf32>, tensor<2x2x34x1024xf32>) -> tensor<2x2x34x1024xf32>
    return %5 : tensor<2x2x34x1024xf32>
  }
}

module {
  func.func @dot_general_with_bias_4(%arg0: tensor<1x1024xf32>, %arg1: tensor<1024x1024xf32>, %bias: tensor<1024x32x2xf32>) -> tensor<1024x32x2xf32> {
    // CHECK: func.func @dot_general_with_bias_4
    // CHECK: "ttir.matmul"
    // CHECK: "ttir.reshape"
    // CHECK: "ttir.add"
    // CHECK-NOT: "ttir.linear"
    %0 = ttir.empty() : tensor<68x1024xf32>
    %1 = "ttir.dot_general"(%arg0, %arg1) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x1024xf32>, tensor<1024x1024xf32>) -> tensor<1x1024xf32>
    %2 = ttir.empty() : tensor<1024x1x1xf32>
    %3 = "ttir.reshape"(%1, %2) <{shape = [1024 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1024xf32>, tensor<1024x1x1xf32>) -> tensor<1024x1x1xf32>
    %4 = ttir.empty() : tensor<1024x32x2xf32>
    %5 = "ttir.add"(%3, %bias, %4) : (tensor<1024x1x1xf32>, tensor<1024x32x2xf32>, tensor<1024x32x2xf32>) -> tensor<1024x32x2xf32>
    return %5 : tensor<1024x32x2xf32>
  }
}

// The following would be a positive case if bias was broadcasted and reshaped from [1024].
module{
  func.func @dot_general_with_bias_5(%arg0: tensor<68x1024xf32>, %arg1: tensor<1024x1024xf32>, %arg2: tensor<2x34x1024xf32>) -> tensor<2x34x1024xf32> {
    // CHECK: func.func @dot_general_with_bias_5
    // CHECK: "ttir.matmul"(%arg0, %arg1, %0)
    // CHECK-NOT: "ttir.linear"
    %0 = ttir.empty() : tensor<68x1024xf32>
    %1 = "ttir.matmul"(%arg0, %arg1, %0) <{transpose_a = false, transpose_b = false}> : (tensor<68x1024xf32>, tensor<1024x1024xf32>, tensor<68x1024xf32>) -> tensor<68x1024xf32>
    %2 = ttir.empty() : tensor<2x34x1024xf32>
    %3 = "ttir.reshape"(%1, %2) <{shape = [2 : i32, 34 : i32, 1024 : i32]}> : (tensor<68x1024xf32>, tensor<2x34x1024xf32>) -> tensor<2x34x1024xf32>
    %4 = ttir.empty() : tensor<2x34x1024xf32>
    %5 = "ttir.add"(%3, %arg2, %4) : (tensor<2x34x1024xf32>, tensor<2x34x1024xf32>, tensor<2x34x1024xf32>) -> tensor<2x34x1024xf32>
    return %5 : tensor<2x34x1024xf32>
  }
}
