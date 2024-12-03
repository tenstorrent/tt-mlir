// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module @jit_iota attributes {} {
  func.func public @test_iota() -> tensor<1x32x128x128xf32> {
    // CHECK: %[[C:.*]] = "ttir.arange"[[C:.*]]
    %0 = "stablehlo.iota"() {iota_dimension = 1: i64} : () -> tensor<1x32x128x128xf32>
    return %0 : tensor<1x32x128x128xf32>
  }
}
