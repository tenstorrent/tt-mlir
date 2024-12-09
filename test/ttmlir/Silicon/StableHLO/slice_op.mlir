// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | \
// RUN:     ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
// RUN: FileCheck --input-file=%t.mlir %s

module @jit_eltwise_subtract attributes {} {
  func.func public @test_slice(%arg0: tensor<32x64xbf16>) -> tensor<8x8xbf16> {
    // CHECK-LABEL: func.func public @test_slice
    // CHECK: ttnn.empty
    // CHECK: ttnn.slice
    // CHECK-SAME: begins = [0 : i32, 16 : i32],
    // CHECK-SAME: ends = [16 : i32, 32 : i32],
    // CHECK-SAME: step = [2 : i32, 2 : i32]
    // CHECK-SAME: tensor<32x64xbf16,
    // CHECK-SAME: tensor<8x8xbf16,
    // CHECK-SAME: -> tensor<8x8xbf16
    %result = "stablehlo.slice"(%arg0) {
      start_indices = array<i64: 0, 16>,
      limit_indices = array<i64: 16, 32>,
      strides = array<i64: 2, 2>
    } : (tensor<32x64xbf16>) -> tensor<8x8xbf16>
    return %result : tensor<8x8xbf16>
  }
}
