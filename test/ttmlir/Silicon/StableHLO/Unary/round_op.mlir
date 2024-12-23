// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | \
// RUN:     ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
// RU1N: FileCheck --input-file=%t.mlir %s

module @jit_gather attributes {} {
    func.func public @test_round(%arg0: tensor<4xbf16>) -> tensor<4xbf16> {
        // CHECK-LABEL: func.func public @test_round
        // CHECK: ttnn.empty
        // CHECK: ttnn.round
        // CHECK-SAME: tensor<4xbf16>
        // CHECK-SAME: tensor<4xbf16>
        // CHECK-SAME: tensor<4xbf16>
        %0 = stablehlo.round_nearest_afz %arg0 : tensor<4xbf16>
        return %0 : tensor<4xbf16>
  }
  func.func public @test_roundnearesteven(%arg0: tensor<4xbf16>) -> tensor<4xbf16> {
        // CHECK-LABEL: func.func public @test_roundnearesteven
        // CHECK: ttnn.empty
        // CHECK: ttnn.round
        // CHECK-SAME: tensor<4xbf16>
        // CHECK-SAME: tensor<4xbf16>
        // CHECK-SAME: tensor<4xbf16>
        %0 = stablehlo.round_nearest_even %arg0 : tensor<4xbf16>
        return %0 : tensor<4xbf16>
  }
}
