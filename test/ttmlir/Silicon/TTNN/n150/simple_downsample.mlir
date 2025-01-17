// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

module {
  func.func @downsample2d_scale_unifrom(%arg0: tensor<4x64x128x3xbf16>) -> tensor<4x32x64x3xbf16> {
    %0 = tensor.empty() : tensor<4x32x64x3xbf16>
    // CHECK: "ttnn.downsample"
    %1 = "ttir.downsample2d"(%arg0, %0) <{scale_factor = 2 : si32}> : (tensor<4x64x128x3xbf16>, tensor<4x32x64x3xbf16>) -> tensor<4x32x64x3xbf16>
    return %1 : tensor<4x32x64x3xbf16>
  }
}
