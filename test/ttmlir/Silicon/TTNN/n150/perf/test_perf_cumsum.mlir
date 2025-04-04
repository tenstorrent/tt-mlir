// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

module @moreh_cumsum attributes {} {
  func.func public @test_moreh_cumsum_dim0(%arg0: tensor<1x32x128x128xbf16>) -> tensor<1x32x128x128xbf16> {
    // CHECK-LABEL: func.func public @test_moreh_cumsum_dim0
    %0 = ttir.empty() : tensor<1x32x128x128xbf16>
    // CHECK: ttnn.moreh_cumsum
    // CHECK-SAME: dim = 0 : i64
    // CHECK-SAME: tensor<1x32x128x128xbf16,
    // CHECK-SAME: -> tensor<1x32x128x128xbf16,
    %1 = "ttir.cumsum"(%arg0, %0) <{dim = 0 : i64}> : (tensor<1x32x128x128xbf16>, tensor<1x32x128x128xbf16>) -> tensor<1x32x128x128xbf16>
    return %1 : tensor<1x32x128x128xbf16>
  }
}
