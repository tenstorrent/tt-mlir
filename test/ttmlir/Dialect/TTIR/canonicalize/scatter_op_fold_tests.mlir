// RUN: ttmlir-opt --canonicalize %s | FileCheck %s

module {
  func.func @scatter_empty_source_index_fold(%arg0: tensor<32x32xf32>, %arg1: tensor<0x32xi32>, %arg2: tensor<0x32xf32>) -> tensor<32x32xf32> {
    // CHECK-LABEL: func.func @scatter_empty_source_index_fold
    // CHECK-NOT: ttir.scatter
    // CHECK: return %arg0
    %0 = "ttir.scatter"(%arg0, %arg1, %arg2) <{dim = 0 : i32, scatter_reduce_type = #ttcore.reduce_type<invalid>}> : (tensor<32x32xf32>, tensor<0x32xi32>, tensor<0x32xf32>) -> tensor<32x32xf32>
    return %0 : tensor<32x32xf32>
  }

  func.func @scatter_no_fold(%arg0: tensor<32x32xf32>, %arg1: tensor<16x32xi32>, %arg2: tensor<16x32xf32>) -> tensor<32x32xf32> {
    // CHECK-LABEL: func.func @scatter_no_fold
    // CHECK: ttir.scatter
    %0 = "ttir.scatter"(%arg0, %arg1, %arg2) <{dim = 0 : i32, scatter_reduce_type = #ttcore.reduce_type<invalid>}> : (tensor<32x32xf32>, tensor<16x32xi32>, tensor<16x32xf32>) -> tensor<32x32xf32>
    return %0 : tensor<32x32xf32>
  }
}
