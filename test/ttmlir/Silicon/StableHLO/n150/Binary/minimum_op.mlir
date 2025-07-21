// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

module @jit_eltwise_minimum attributes {} {
  func.func public @test_minimum(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    // CHECK-LABEL: func.func public @test_minimum
    // CHECK: ttnn.minimum
    // CHECK-SAME: tensor<64x128xf32,
    // CHECK-SAME: tensor<64x128xf32,
    // CHECK-SAME: -> tensor<64x128xf32,
    %0 = stablehlo.minimum %arg0, %arg1 : tensor<64x128xf32>
    return %0 : tensor<64x128xf32>
  }
}
