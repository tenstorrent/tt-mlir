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
    // CHECK: %[[VAL:[0-9]+]] = "ttnn.transpose"
    // CHECK-SAME: {dim0 = 3 : si32, dim1 = 2 : si32}
    // CHECK-SAME: tensor<1x32x64x128xf32,
    // CHECK-SAME: -> tensor<1x32x128x64xf32
    // CHECK: "ttnn.transpose"(%[[VAL]])
    // CHECK-SAME: {dim0 = 2 : si32, dim1 = 1 : si32}
    // CHECK-SAME: tensor<1x32x128x64xf32,
    // CHECK-SAME: -> tensor<1x128x32x64xf32,
    %0 = stablehlo.transpose %arg0, dims = [0, 3, 1, 2] : (tensor<1x32x64x128xf32>) -> tensor<1x128x32x64xf32>
    return %0 : tensor<1x128x32x64xf32>
  }
}
