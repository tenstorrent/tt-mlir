// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module @jit_transpose attributes {} {
  func.func public @test_transpose(%arg0: tensor<64x128xf32>) -> tensor<128x64xf32> {
    %0 = stablehlo.transpose %arg0, dims = [1,0] : (tensor<64x128xf32>) -> tensor<128x64xf32>
    // CHECK: %[[C:.*]] = tensor.empty[[C:.*]]
    // CHECK: %[[C:.*]] = "ttir.transpose"[[C:.*]]
    return %0 : tensor<128x64xf32>
  }
}
