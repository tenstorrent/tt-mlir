// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module attributes {} {
  func.func @test_empty_int8() -> tensor<12xi32> {
    %0 = "ttir.constant"() <{value = dense<[[[0, 1], [2, 3], [4, 5]], [[6, 7], [8, 9], [10, 11]]]> : tensor<2x3x2xi32>}> : () -> tensor<2x3x2xi32>
    // CHECK: "ttnn.constant"() <{value = dense<[0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11]> : tensor<12xi32>}>
    %1 = tensor.empty() : tensor<2x2x3xi32>
    %2 = "ttir.permute"(%0, %1) <{permutation = array<i64: 2, 0, 1>}> : (tensor<2x3x2xi32>, tensor<2x2x3xi32>) -> tensor<2x2x3xi32>
    %3 = tensor.empty() : tensor<12xi32>
    %4 = "ttir.reshape"(%2, %3) <{shape = [12: i32]}> : (tensor<2x2x3xi32>, tensor<12xi32>) -> tensor<12xi32>
    return %4 : tensor<12xi32>
  }
}
