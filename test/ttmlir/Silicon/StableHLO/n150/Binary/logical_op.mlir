// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | \
// RUN:     ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
// RUN: FileCheck --input-file=%t.mlir %s

module @jit_eltwise_compare attributes {} {
  func.func public @logical_and(%arg0: tensor<64x128xi1>, %arg1: tensor<64x128xi1>) -> tensor<64x128xi1> {
    // CHECK-LABEL: func.func public @logical_and
    // CHECK: ttnn.logical_and
    // CHECK-SAME: tensor<64x128xbf16,
    // CHECK-SAME: tensor<64x128xbf16,
    // CHECK-SAME: -> tensor<64x128xbf16,
    %0 = stablehlo.and  %arg0, %arg1 : tensor<64x128xi1>
    return %0 : tensor<64x128xi1>
  }

  func.func public @logical_or(%arg0: tensor<64x128xi1>, %arg1: tensor<64x128xi1>) -> tensor<64x128xi1> {
    // CHECK-LABEL: func.func public @logical_or
    // CHECK: ttnn.logical_or
    // CHECK-SAME: tensor<64x128xbf16,
    // CHECK-SAME: tensor<64x128xbf16,
    // CHECK-SAME: -> tensor<64x128xbf16,
    %0 = stablehlo.or  %arg0, %arg1 : tensor<64x128xi1>
    return %0 : tensor<64x128xi1>
  }

  func.func public @logical_xor(%arg0: tensor<64x128xi1>, %arg1: tensor<64x128xi1>) -> tensor<64x128xi1> {
    // CHECK-LABEL: func.func public @logical_xor
    // CHECK: ttnn.logical_xor
    // CHECK-SAME: tensor<64x128xbf16,
    // CHECK-SAME: tensor<64x128xbf16,
    // CHECK-SAME: -> tensor<64x128xbf16,
    %0 = stablehlo.xor  %arg0, %arg1 : tensor<64x128xi1>
    return %0 : tensor<64x128xi1>
  }
}
