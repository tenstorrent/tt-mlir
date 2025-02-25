// RUN: ttmlir-opt %s | FileCheck %s

module attributes {} {
  func.func @reverse_first_dim(%arg0: tensor<32x64xf32>) -> tensor<32x64xf32> {
    %0 = tensor.empty() : tensor<32x64xf32>
    // CHECK: = "ttir.reverse"
    %1 = "ttir.reverse"(%arg0, %0) <{dimensions = array<i64: 0>}> : (tensor<32x64xf32>, tensor<32x64xf32>) -> tensor<32x64xf32>
    return %1 : tensor<32x64xf32>
  }

  func.func @reverse_second_dim(%arg0: tensor<32x64xf32>) -> tensor<32x64xf32> {
    %0 = tensor.empty() : tensor<32x64xf32>
    // CHECK: = "ttir.reverse"
    %1 = "ttir.reverse"(%arg0, %0) <{dimensions = array<i64: 1>}> : (tensor<32x64xf32>, tensor<32x64xf32>) -> tensor<32x64xf32>
    return %1 : tensor<32x64xf32>
  }

  func.func @reverse_both_dims(%arg0: tensor<32x64xf32>) -> tensor<32x64xf32> {
    %0 = tensor.empty() : tensor<32x64xf32>
    // CHECK: = "ttir.reverse"
    %1 = "ttir.reverse"(%arg0, %0) <{dimensions = array<i64: 0, 1>}> : (tensor<32x64xf32>, tensor<32x64xf32>) -> tensor<32x64xf32>
    return %1 : tensor<32x64xf32>
  }
}
