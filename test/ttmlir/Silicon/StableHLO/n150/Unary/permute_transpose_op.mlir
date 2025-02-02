// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | \
// RUN:     ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
// RUN: FileCheck --input-file=%t.mlir %s

module {
  func.func public @test_permute_transpose(%arg0: tensor<1x32x64x128xf32>) -> tensor<1x128x32x64xf32> {
    // CHECK-LABEL: func.func public @test_permute_transpose
    // CHECK: %[[VAL:[0-9]+]] = "ttnn.permute"
    // CHECK-SAME: permutation = array<i64: 0, 3, 1, 2>
    // CHECK-SAME: tensor<1x32x64x128xf32,
    // CHECK-SAME: -> tensor<1x128x32x64xf32,
    %0 = stablehlo.transpose %arg0, dims = [0, 3, 1, 2] : (tensor<1x32x64x128xf32>) -> tensor<1x128x32x64xf32>
    return %0 : tensor<1x128x32x64xf32>
  }
}
