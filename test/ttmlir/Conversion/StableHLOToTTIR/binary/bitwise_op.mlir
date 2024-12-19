// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s

module @jit_eltwise_bitwise attributes {} {
  func.func public @bitwise_and(%arg0: tensor<32x32xi32>, %arg1: tensor<32x32xi32>) -> tensor<32x32xi32> {
    %0 = stablehlo.and %arg0, %arg1 : tensor<32x32xi32>
    // CHECK: %[[EMPTY:[0-9]+]] = tensor.empty() : tensor<32x32xi32>
    // CHECK: %[[AND:[0-9]+]] = "ttir.bitwise_and"(%arg0, %arg1, %[[EMPTY]]){{.*}} -> tensor<32x32xi32>
    return %0 : tensor<32x32xi32>
    // CHECK: return %[[AND]] : tensor<32x32xi32>
  }

  func.func public @bitwise_or(%arg0: tensor<32x32xi32>, %arg1: tensor<32x32xi32>) -> tensor<32x32xi32> {
    %0 = stablehlo.or %arg0, %arg1 : tensor<32x32xi32>
    // CHECK: %[[EMPTY:[0-9]+]] = tensor.empty() : tensor<32x32xi32>
    // CHECK: %[[OR:[0-9]+]] = "ttir.bitwise_or"(%arg0, %arg1, %[[EMPTY]]){{.*}} -> tensor<32x32xi32>
    return %0 : tensor<32x32xi32>
    // CHECK: return %[[OR]] : tensor<32x32xi32>
  }

  func.func public @bitwise_xor(%arg0: tensor<32x32xi32>, %arg1: tensor<32x32xi32>) -> tensor<32x32xi32> {
    %0 = stablehlo.xor %arg0, %arg1 : tensor<32x32xi32>
    // CHECK: %[[EMPTY:[0-9]+]] = tensor.empty() : tensor<32x32xi32>
    // CHECK: %[[XOR:[0-9]+]] = "ttir.bitwise_xor"(%arg0, %arg1, %[[EMPTY]]){{.*}} -> tensor<32x32xi32>
    return %0 : tensor<32x32xi32>
    // CHECK: return %[[XOR]] : tensor<32x32xi32>
  }

  func.func public @bitwise_not(%arg0: tensor<32x32xi32>) -> tensor<32x32xi32> {
    %0 = stablehlo.not %arg0 : tensor<32x32xi32>
    // CHECK: %[[EMPTY:[0-9]+]] = tensor.empty() : tensor<32x32xi32>
    // CHECK: %[[NOT:[0-9]+]] = "ttir.bitwise_not"(%arg0, %[[EMPTY]]){{.*}} -> tensor<32x32xi32>
    return %0 : tensor<32x32xi32>
    // CHECK: return %[[NOT]] : tensor<32x32xi32>
  }
}
