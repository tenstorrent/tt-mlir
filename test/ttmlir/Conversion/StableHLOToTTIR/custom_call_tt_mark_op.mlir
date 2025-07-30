// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
module @jit__lambda_ attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32, tt.mark_function_defined = true} {
  func.func private @tt.mark(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }
  func.func public @main(%arg0: tensor<4xf32>, %arg1: tensor<3x4xf32>, %arg2: tensor<1x3xf32>) -> (tensor<1x4xf32> {jax.result_info = "result"}) {
    // CHECK-LABEL: func.func public @main
    // CHECK-SAME: %[[ARG0:.*]]: tensor<4xf32> {tt.role = "weight"}
    // CHECK-SAME: %[[ARG1:.*]]: tensor<3x4xf32> {tt.role = "weight"}
    // CHECK-SAME: %[[ARG2:.*]]: tensor<1x3xf32>
    %0 = stablehlo.custom_call @tt.mark(%arg0) {tt.role = "weight"} : (tensor<4xf32>) -> tensor<4xf32>
    %1 = stablehlo.custom_call @tt.mark(%arg1) {tt.role = "weight"} : (tensor<3x4xf32>) -> tensor<3x4xf32>
    %2 = stablehlo.dot_general %arg2, %1, contracting_dims = [1] x [0] : (tensor<1x3xf32>, tensor<3x4xf32>) -> tensor<1x4xf32>
    // CHECK: %[[DOT:.*]] = "ttir.dot_general"(%[[ARG2]], %[[ARG1]]) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<1x3xf32>, tensor<3x4xf32>) -> tensor<1x4xf32>
    %3 = stablehlo.reshape %0 : (tensor<4xf32>) -> tensor<1x4xf32>
    // CHECK: %[[EMPTY:.*]] = ttir.empty() : tensor<1x4xf32>
    // CHECK: %[[RESHAPE:.*]] = "ttir.reshape"(%[[ARG0]], %[[EMPTY]]) <{shape = [1 : i32, 4 : i32]}> : (tensor<4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
    %4 = stablehlo.add %2, %3 : tensor<1x4xf32>
    // CHECK: %[[EMPTY2:.*]] = ttir.empty() : tensor<1x4xf32>
    // CHECK: %[[ADD:.*]] = "ttir.add"(%[[DOT]], %[[RESHAPE]], %[[EMPTY2]]) : (tensor<1x4xf32>, tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
    return %4 : tensor<1x4xf32>
    // CHECK: return %[[ADD]] : tensor<1x4xf32>
  }
}