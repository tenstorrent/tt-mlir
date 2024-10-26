// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
module @jit_eltwise_compare attributes {} {
  func.func public @logical_and(%arg0: tensor<13x31xi1>, %arg1: tensor<13x31xi1>) -> tensor<13x31xi1> {
    %0 = stablehlo.and  %arg0, %arg1 : tensor<13x31xi1>
    // CHECK: %[[E:.*]] = tensor.empty() : tensor<13x31xi32>
    // CHECK: = "ttir.logical_and"(%arg0, %arg1, %[[E]])
    // CHECK-SAME: (tensor<13x31xi32>, tensor<13x31xi32>, tensor<13x31xi32>) -> tensor<13x31xi32>
    return %0 : tensor<13x31xi1>
    // CHECK: return %1 : tensor<13x31xi32>
  }

  func.func public @logical_or(%arg0: tensor<13x31xi1>, %arg1: tensor<13x31xi1>) -> tensor<13x31xi1> {
    %0 = stablehlo.or  %arg0, %arg1 : tensor<13x31xi1>
    // CHECK: %[[E:.*]] = tensor.empty() : tensor<13x31xi32>
    // CHECK: = "ttir.logical_or"(%arg0, %arg1, %[[E]])
    // CHECK-SAME: (tensor<13x31xi32>, tensor<13x31xi32>, tensor<13x31xi32>) -> tensor<13x31xi32>
    return %0 : tensor<13x31xi1>
    // CHECK: return %1 : tensor<13x31xi32>
  }

func.func public @logical_not(%arg0: tensor<13x31xi1>) -> tensor<13x31xi1> {
    %0 = stablehlo.not  %arg0 : tensor<13x31xi1>
    // CHECK: %[[E:.*]] = tensor.empty() : tensor<13x31xi32>
    // CHECK: = "ttir.logical_not"(%arg0, %[[E]])
    // CHECK-SAME: (tensor<13x31xi32>, tensor<13x31xi32>) -> tensor<13x31xi32>
    return %0 : tensor<13x31xi1>
    // CHECK: return %1 : tensor<13x31xi32>
  }

func.func public @logical_not_scalar(%arg0: tensor<i1>) -> tensor<i1> {
    %0 = stablehlo.not  %arg0 : tensor<i1>
    // CHECK: %[[E:.*]] = tensor.empty() : tensor<1xi32>
    // CHECK: = "ttir.logical_not"(%arg0, %[[E]])
    // CHECK-SAME: (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    return %0 : tensor<i1>
    // CHECK: return %1 : tensor<1xi32>
  }
}
