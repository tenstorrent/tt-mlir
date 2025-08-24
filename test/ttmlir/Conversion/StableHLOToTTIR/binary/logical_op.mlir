// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module @jit_eltwise_logical attributes {} {
  func.func public @logical_and(%arg0: tensor<32x32xi1>, %arg1: tensor<32x32xi1>) -> tensor<32x32xi1> {
    // CHECK: %[[E:.*]] = ttir.empty() : [[TENSOR:tensor<32x32xi1>]]
    // CHECK: = "ttir.logical_and"(%arg0, %arg1, %[[E]])
    // CHECK-SAME: ([[TENSOR]], [[TENSOR]], [[TENSOR]]) -> [[TENSOR]]
    %0 = stablehlo.and  %arg0, %arg1 : tensor<32x32xi1>
    // CHECK: return %1 : [[TENSOR]]
    return %0 : tensor<32x32xi1>
  }

  func.func public @logical_or(%arg0: tensor<32x32xi1>, %arg1: tensor<32x32xi1>) -> tensor<32x32xi1> {
    // CHECK: %[[E:.*]] = ttir.empty() : [[TENSOR:tensor<32x32xi1>]]
    // CHECK: = "ttir.logical_or"(%arg0, %arg1, %[[E]])
    // CHECK-SAME: ([[TENSOR]], [[TENSOR]], [[TENSOR]]) -> [[TENSOR]]
    %0 = stablehlo.or  %arg0, %arg1 : tensor<32x32xi1>
    // CHECK: return %1 : [[TENSOR]]
    return %0 : tensor<32x32xi1>
  }

  func.func public @logical_xor(%arg0: tensor<32x32xi1>, %arg1: tensor<32x32xi1>) -> tensor<32x32xi1> {
    // CHECK: %[[E:.*]] = ttir.empty() : [[TENSOR:tensor<32x32xi1>]]
    // CHECK: = "ttir.logical_xor"(%arg0, %arg1, %[[E]])
    // CHECK-SAME: ([[TENSOR]], [[TENSOR]], [[TENSOR]]) -> [[TENSOR]]
    %0 = stablehlo.xor  %arg0, %arg1 : tensor<32x32xi1>
    // CHECK: return %1 : [[TENSOR]]
    return %0 : tensor<32x32xi1>
  }
}
