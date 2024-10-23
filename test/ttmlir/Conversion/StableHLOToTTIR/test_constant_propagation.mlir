// UNSUPPORTED: true
// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline  %s | FileCheck %s
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module @jit_eltwise_convert attributes {} {
  func.func public @test_convert(%arg0: tensor<32x32xi32>) -> tensor<32x32xi32> {
    %0 = stablehlo.constant dense<0> : tensor<32x32xi32>
    %1 = stablehlo.add %0, %0 : tensor<32x32xi32>
    %2 = stablehlo.multiply %1, %arg0 : tensor<32x32xi32>
    return %2 : tensor<32x32xi32>
    // CHECK-NOT: %[[C:.*]] = "ttir.add"[[C:.*]]
  }
}
