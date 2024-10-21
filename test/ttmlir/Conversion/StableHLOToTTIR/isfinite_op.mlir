// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module @jit_eltwise_isfinite attributes {} {
  func.func public @test_isfinite(%arg0: tensor<32x32x3xf32>) -> tensor<32x32x3xi1> {
    // CHECK: %[[E:.*]] = tensor.empty() : tensor<32x32x3xbf16>
    // CHECK: %[[C:.*]] = "ttir.isfinite"(%arg0, %[[E]])
    // CHECK-SAME: (tensor<32x32x3xf32>, tensor<32x32x3xbf16>) -> tensor<32x32x3xbf16>
    %0 = stablehlo.is_finite %arg0 : (tensor<32x32x3xf32>) -> tensor<32x32x3xi1>
    // CHECK: return %[[C]] : tensor<32x32x3xbf16>
    return %0 : tensor<32x32x3xi1>
  }
}
