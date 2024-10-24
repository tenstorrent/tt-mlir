// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module @jit_eltwise_expm1 attributes {} {
  func.func public @test_expm1(%arg0: tensor<13x21x3xf32>) -> tensor<13x21x3xf32> {
    %0 = stablehlo.exponential_minus_one %arg0 : tensor<13x21x3xf32>
    // CHECK: %[[C:.*]] = tensor.empty[[C:.*]]
    // CHECK: %[[C:.*]] = "ttir.expm1"[[C:.*]]
    return %0 : tensor<13x21x3xf32>
  }
}
