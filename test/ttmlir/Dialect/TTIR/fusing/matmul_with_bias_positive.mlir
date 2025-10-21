// RUN: ttmlir-opt -ttir-to-ttir-decomposition -ttir-fusing -o %t %s
// RUN: FileCheck %s --input-file=%t

// ===----------------------------------------------------------------------===
// POSITIVE CASES: Operations that SHOULD be fused into linear op with bias
// ===----------------------------------------------------------------------===

module {
  func.func @dot_general_with_bias_1(%arg0: tensor<68x1024xf32>, %arg1: tensor<1024x1024xf32>, %bias: tensor<68x1024xf32>) -> tensor<2x34x16x64xf32> {
    // CHECK: func.func @dot_general_with_bias_1
    // CHECK: "ttir.linear"(%arg0, %arg1, %arg2, %0)
    // CHECK-NOT: "ttir.dot_general"
    // CHECK-NOT: "ttir.matmul"
    // CHECK-NOT: "ttir.add"
    // CHECK: "ttir.reshape"(%1, %2)
    %0 = ttir.empty() : tensor<68x1024xf32>
    %1 = "ttir.dot_general"(%arg0, %arg1) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<68x1024xf32>, tensor<1024x1024xf32>) -> (tensor<68x1024xf32>)
    %2 = ttir.empty() : tensor<68x1024xf32>
    %3 = "ttir.add"(%1, %bias, %2) : (tensor<68x1024xf32>, tensor<68x1024xf32>, tensor<68x1024xf32>) -> tensor<68x1024xf32>
    %4 = ttir.empty() : tensor<2x34x16x64xf32>
    %5 = "ttir.reshape"(%3, %4)<{shape = [2 : i32, 34 : i32, 16 : i32, 64 : i32]}> : (tensor<68x1024xf32>, tensor<2x34x16x64xf32>) -> tensor<2x34x16x64xf32>
    return %5 : tensor<2x34x16x64xf32>
  }
}

module {
  // replace order of operands for add op
  func.func @dot_general_with_bias_2(%arg0: tensor<68x1024xf32>, %arg1: tensor<1024x1024xf32>, %bias: tensor<68x1024xf32>) -> tensor<2x34x16x64xf32> {
    // CHECK: func.func @dot_general_with_bias_2
    // CHECK: "ttir.linear"(%arg0, %arg1, %arg2, %0)
    // CHECK-NOT: "ttir.dot_general"
    // CHECK-NOT: "ttir.matmul"
    // CHECK-NOT: "ttir.add"
    %0 = ttir.empty() : tensor<68x1024xf32>
    %1 = "ttir.dot_general"(%arg0, %arg1) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<68x1024xf32>, tensor<1024x1024xf32>) -> tensor<68x1024xf32>
    %2 = ttir.empty() : tensor<68x1024xf32>
    %3 = "ttir.add"(%bias, %1, %2) : (tensor<68x1024xf32>, tensor<68x1024xf32>, tensor<68x1024xf32>) -> tensor<68x1024xf32>
    %4 = ttir.empty() : tensor<2x34x16x64xf32>
    %5 = "ttir.reshape"(%3, %4)<{shape = [2 : i32, 34 : i32, 16 : i32, 64 : i32]}> : (tensor<68x1024xf32>, tensor<2x34x16x64xf32>) -> tensor<2x34x16x64xf32>
    return %5 : tensor<2x34x16x64xf32>
  }
}

module {
  // dot_general op followed by reshape op before add op
  func.func @dot_general_with_bias_3(%arg0: tensor<68x1024xf32>, %arg1: tensor<1024x1024xf32>, %bias: tensor<2x34x1024xf32>) -> tensor<2x34x1024xf32> {
    // CHECK: func.func @dot_general_with_bias_3
    // CHECK: "ttir.linear"(%arg0, %arg1, %1, %2)
    // CHECK-NOT: "ttir.dot_general"
    // CHECK-NOT: "ttir.matmul"
    // CHECK-NOT: "ttir.add"
    %0 = ttir.empty() : tensor<68x1024xf32>
    %1 = "ttir.dot_general"(%arg0, %arg1) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<68x1024xf32>, tensor<1024x1024xf32>) -> tensor<68x1024xf32>
    %2 = ttir.empty() : tensor<2x34x1024xf32>
    %3 = "ttir.reshape"(%1, %2) <{shape = [2 : i32, 34 : i32, 1024 : i32]}> : (tensor<68x1024xf32>, tensor<2x34x1024xf32>) -> tensor<2x34x1024xf32>
    %4 = ttir.empty() : tensor<2x34x1024xf32>
    %5 = "ttir.add"(%3, %bias, %4) : (tensor<2x34x1024xf32>, tensor<2x34x1024xf32>, tensor<2x34x1024xf32>) -> tensor<2x34x1024xf32>
    return %5 : tensor<2x34x1024xf32>
  }
}
