// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
module @jit_eltwise_logical attributes {} {
  func.func public @logical_not(%arg0: tensor<32x32xi1>) -> tensor<32x32xi1> {
    // CHECK: %[[E:.*]] = tensor.empty() : [[TENSOR:tensor<32x32xbf16>]]
    // CHECK: = "ttir.logical_not"(%arg0, %[[E]])
    // CHECK-SAME: ([[TENSOR]], [[TENSOR]]) -> [[TENSOR]]
    %0 = stablehlo.not  %arg0 : tensor<32x32xi1>
    // CHECK: return %1 : [[TENSOR]]
    return %0 : tensor<32x32xi1>
  }

  func.func public @logical_not_scalar(%arg0: tensor<i1>) -> tensor<i1> {
    // CHECK: %[[E:.*]] = tensor.empty() : [[TENSOR:tensor<1xbf16>]]
    // CHECK: = "ttir.logical_not"(%arg0, %[[E]])
    // CHECK-SAME: ([[TENSOR]], [[TENSOR]]) -> [[TENSOR]]
    %0 = stablehlo.not  %arg0 : tensor<i1>
    // CHECK: return %1 : [[TENSOR]]
    return %0 : tensor<i1>
  }
}
