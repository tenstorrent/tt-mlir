// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module @jit_eltwise_floor attributes {} {
  func.func public @test_floor(%arg0: tensor<32x32x3xf32>) -> tensor<32x32x3xf32> {
    %0 = stablehlo.floor %arg0 : tensor<32x32x3xf32>
    // CHECK: %[[C:.*]] = tensor.empty[[C:.*]]
    // CHECK: %[[C:.*]] = "ttir.floor"[[C:.*]]
    return %0 : tensor<32x32x3xf32>
  }
}
