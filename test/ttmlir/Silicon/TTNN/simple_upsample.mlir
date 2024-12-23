// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

module {
  func.func @upsample_scale_unifrom(%arg0: tensor<4x32x64x3xbf16>) -> tensor<4x64x128x3xbf16> {
    %0 = tensor.empty() : tensor<4x64x128x3xbf16>
    // CHECK: "ttnn.upsample"
    // CHECK-SAME: tensor<4x32x64x3xbf16
    // CHECK-SAME: tensor<4x64x128x3xbf16
    %1 = "ttir.upsample"(%arg0, %0) <{scale_factor = 2 : si32}> : (tensor<4x32x64x3xbf16>, tensor<4x64x128x3xbf16>) -> tensor<4x64x128x3xbf16>
    return %1 : tensor<4x64x128x3xbf16>
  }

  func.func @upsample_scale_nonunifrom(%arg0: tensor<4x32x64x3xbf16>) -> tensor<4x64x64x3xbf16> {
    %0 = tensor.empty() : tensor<4x64x64x3xbf16>
    // CHECK: "ttnn.upsample"
    // CHECK-SAME: tensor<4x32x64x3xbf16
    // CHECK-SAME: tensor<4x64x64x3xbf16
    %1 = "ttir.upsample"(%arg0, %0) <{scale_factor = array<i32: 2, 1>}> : (tensor<4x32x64x3xbf16>, tensor<4x64x64x3xbf16>) -> tensor<4x64x64x3xbf16>
    return %1 : tensor<4x64x64x3xbf16>
  }
}
