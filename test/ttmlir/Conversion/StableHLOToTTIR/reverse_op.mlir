// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s

module @jit_eltwise_reverse attributes {} {
  func.func @reverse_op(%arg0: tensor<32x64xf32>) -> tensor<32x64xf32> {
    %0 = "stablehlo.reverse"(%arg0) {dimensions = array<i64: 1, 0>} : (tensor<32x64xf32>) -> tensor<32x64xf32>
    // CHECK: %[[EMPTY:[0-9]+]] = tensor.empty() : tensor<32x64xf32>
    // CHECK: %[[REV:[0-9]+]] = "ttir.reverse"(%arg0, %0) <{dimensions = array<i64: 1, 0>}> : (tensor<32x64xf32>, tensor<32x64xf32>) -> tensor<32x64xf32>
    return %0 : tensor<32x64xf32>
    // CHECK: return %[[REV]] : tensor<32x64xf32>
  }
}
