// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
module @jit_eltwise_compare attributes {} {
  func.func public @test_eq(%arg0: tensor<13x31xf32>, %arg1: tensor<13x31xf32>) -> tensor<13x31xi1> {
    %0 = stablehlo.compare  EQ, %arg0, %arg1 : (tensor<13x31xf32>, tensor<13x31xf32>) -> tensor<13x31xi1>
    // CHECK: %[[E:.*]] = tensor.empty() : tensor<13x31xi32>
    // CHECK: = "ttir.eq"(%arg0, %arg1, %[[E]])
    // CHECK-SAME: (tensor<13x31xf32>, tensor<13x31xf32>, tensor<13x31xi32>) -> tensor<13x31xi32>
    return %0 : tensor<13x31xi1>
    // CHECK: return %1 : tensor<13x31xi32>
  }

  func.func public @test_ne(%arg0: tensor<13x31xf32>, %arg1: tensor<13x31xf32>) -> tensor<13x31xi1> {
    %0 = stablehlo.compare  NE, %arg0, %arg1 : (tensor<13x31xf32>, tensor<13x31xf32>) -> tensor<13x31xi1>
    // CHECK: %[[E:.*]] = tensor.empty() : tensor<13x31xi32>
    // CHECK: = "ttir.ne"(%arg0, %arg1, %[[E]])
    // CHECK-SAME: (tensor<13x31xf32>, tensor<13x31xf32>, tensor<13x31xi32>) -> tensor<13x31xi32>
    return %0 : tensor<13x31xi1>
    // CHECK: return %1 : tensor<13x31xi32>
  }

  func.func public @test_ge(%arg0: tensor<13x31xf32>, %arg1: tensor<13x31xf32>) -> tensor<13x31xi1> {
    %0 = stablehlo.compare  GE, %arg0, %arg1 : (tensor<13x31xf32>, tensor<13x31xf32>) -> tensor<13x31xi1>
    // CHECK: %[[E:.*]] = tensor.empty() : tensor<13x31xi32>
    // CHECK: = "ttir.ge"(%arg0, %arg1, %[[E]])
    // CHECK-SAME: (tensor<13x31xf32>, tensor<13x31xf32>, tensor<13x31xi32>) -> tensor<13x31xi32>
    return %0 : tensor<13x31xi1>
    // CHECK: return %1 : tensor<13x31xi32>
  }

  func.func public @test_gt(%arg0: tensor<13x31xf32>, %arg1: tensor<13x31xf32>) -> tensor<13x31xi1> {
    %0 = stablehlo.compare  GT, %arg0, %arg1 : (tensor<13x31xf32>, tensor<13x31xf32>) -> tensor<13x31xi1>
    // CHECK: %[[E:.*]] = tensor.empty() : tensor<13x31xi32>
    // CHECK: = "ttir.gt"(%arg0, %arg1, %[[E]])
    // CHECK-SAME: (tensor<13x31xf32>, tensor<13x31xf32>, tensor<13x31xi32>) -> tensor<13x31xi32>
    return %0 : tensor<13x31xi1>
    // CHECK: return %1 : tensor<13x31xi32>
  }

  func.func public @test_le(%arg0: tensor<13x31xf32>, %arg1: tensor<13x31xf32>) -> tensor<13x31xi1> {
    %0 = stablehlo.compare  LE, %arg0, %arg1 : (tensor<13x31xf32>, tensor<13x31xf32>) -> tensor<13x31xi1>
    // CHECK: %[[E:.*]] = tensor.empty() : tensor<13x31xi32>
    // CHECK: = "ttir.le"(%arg0, %arg1, %[[E]])
    // CHECK-SAME: (tensor<13x31xf32>, tensor<13x31xf32>, tensor<13x31xi32>) -> tensor<13x31xi32>
    return %0 : tensor<13x31xi1>
    // CHECK: return %1 : tensor<13x31xi32>
  }

  func.func public @test_lt(%arg0: tensor<13x31xf32>, %arg1: tensor<13x31xf32>) -> tensor<13x31xi1> {
    %0 = stablehlo.compare  LT, %arg0, %arg1 : (tensor<13x31xf32>, tensor<13x31xf32>) -> tensor<13x31xi1>
    // CHECK: %[[E:.*]] = tensor.empty() : tensor<13x31xi32>
    // CHECK: = "ttir.lt"(%arg0, %arg1, %[[E]])
    // CHECK-SAME: (tensor<13x31xf32>, tensor<13x31xf32>, tensor<13x31xi32>) -> tensor<13x31xi32>
    return %0 : tensor<13x31xi1>
    // CHECK: return %1 : tensor<13x31xi32>
  }
}
