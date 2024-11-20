// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | \
// RUN:     ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

module @jit_eltwise_select attributes {} {
  func.func public @test_select(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    // CHECK-LABEL: func.func public @test_select
    // CHECK: tensor.empty
    // CHECK: [[EQ:{{0-9}}+]] = "ttnn.eq"
    // CHECK-SAME: tensor<64x128xf32
    // CHECK-SAME: tensor<64x128xf32
    // CHECK-SAME: tensor<64x128xbf16
    // CHECK-SAME: -> tensor<64x128xbf16
    %0 = stablehlo.compare EQ, %arg0, %arg1 : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xi1>
    // CHECK: ttnn.where
    // CHECK-SAME: [[EQ]]
    // CHECK-SAME: tensor<64x128xbf16
    // CHECK-SAME: tensor<64x128xf32
    // CHECK-SAME: tensor<64x128xf32
    // CHECK-SAME: tensor<64x128xf32
    // CHECK-SAME: -> tensor<64x128xf32
    %1 = stablehlo.select %0, %arg0, %arg1 : (tensor<64x128xi1>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    return %1 : tensor<64x128xf32>
  }
}
