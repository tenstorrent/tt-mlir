// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | \
// RUN:     ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
// RUN: FileCheck --input-file=%t.mlir %s

module @jit_eltwise_convert attributes {} {
  func.func public @test_convert(%arg0: tensor<64x128xbf16>) -> tensor<64x128xf32> {
    // CHECK-LABEL: func.func public @test_convert
    // CHECK: ttnn.typecast
    // CHECK-SAME: tensor<64x128xbf16,
    // CHECK-SAME: tensor<64x128xf32,
    %0 = stablehlo.convert %arg0 : (tensor<64x128xbf16>) -> tensor<64x128xf32>
    return %0 : tensor<64x128xf32>
  }

  func.func public @test_add(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xbf16> {
    // CHECK-LABEL: func.func public @test_add
    // CHECK: ttnn.typecast
    // CHECK-SAME: tensor<64x128xf32,
    // CHECK-SAME: tensor<64x128xbf16,
    %0 = stablehlo.convert %arg0 : (tensor<64x128xf32>) -> tensor<64x128xbf16>
    // CHECK: ttnn.typecast
    // CHECK-SAME: tensor<64x128xf32,
    // CHECK-SAME: tensor<64x128xbf16,
    %1 = stablehlo.convert %arg1 : (tensor<64x128xf32>) -> tensor<64x128xbf16>
    %2 = stablehlo.add %0, %1 : tensor<64x128xbf16>
    return %2 : tensor<64x128xbf16>
  }
}
