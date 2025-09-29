// RUN: ttmlir-opt -ttir-to-ttir-decomposition -ttir-fusing -o %t %s
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
