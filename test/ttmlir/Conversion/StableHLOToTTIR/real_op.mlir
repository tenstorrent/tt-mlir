// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>

module @jit_eltwise_real attributes {} {
  func.func public @test_real(%operand: tensor<2xcomplex<f32>>) -> tensor<2xf32> {
    %result = "stablehlo.real"(%operand) : (tensor<2xcomplex<f32>>) -> tensor<2xf32>
    // CHECK: %[[C:.*]] = tensor.empty[[C:.*]]
    // CHECK: %[[C:.*]] = "ttir.real"[[C:.*]]
    return %result : tensor<2xf32>
  }
}
