// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
module @jit_eltwise_round attributes {} {
  func.func public @test_round(%arg0: tensor<4xbf16>) -> tensor<4xbf16> {
    %0 = stablehlo.round_nearest_afz %arg0 : tensor<4xbf16>
    // CHECK: %[[C:.*]] = tensor.empty[[C:.*]]
    // CHECK: %[[C:.*]] = "ttir.round"[[C:.*]]
    return %0 : tensor<4xbf16>
  }
  func.func public @test_roundnearesteven(%arg0: tensor<4xbf16>) -> tensor<4xbf16> {
    %0 = stablehlo.round_nearest_even %arg0 : tensor<4xbf16>
    // CHECK: %[[C:.*]] = tensor.empty[[C:.*]]
    // CHECK: %[[C:.*]] = "ttir.roundnearesteven"[[C:.*]]
    return %0 : tensor<4xbf16>
  }
}
