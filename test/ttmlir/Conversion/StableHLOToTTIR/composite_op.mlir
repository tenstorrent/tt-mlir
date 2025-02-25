// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
module @jit_eltwise_add attributes {} {
  func.func private @add_impl(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<13x21x3xf32>
    return %0 : tensor<13x21x3xf32>
  }

  func.func public @main(%arg0: tensor<13x21x3xf32>, %arg1: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %results = stablehlo.composite "jit_eltwise_add.my_add" %arg0, %arg1 {
        decomposition = @add_impl
    } : (tensor<13x21x3xf32>, tensor<13x21x3xf32>) -> tensor<13x21x3xf32>
    // CHECK: = call @add_impl
    return %results : tensor<13x21x3xf32>
  }
}
