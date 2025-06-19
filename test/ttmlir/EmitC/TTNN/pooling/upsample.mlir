// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %basename_t.ttnn
// RUN: ttmlir-opt --ttnn-backend-to-emitc-pipeline="target-dylib=true" %t.mlir > %t2.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir > %basename_t.cpp

module {
  func.func @upsample2d_scale_unifrom(%arg0: tensor<4x32x64x3xbf16>) -> tensor<4x64x128x3xbf16> {
    %0 = ttir.empty() : tensor<4x64x128x3xbf16>
    %1 = "ttir.upsample2d"(%arg0, %0) <{scale_factor = 2 : si32}> : (tensor<4x32x64x3xbf16>, tensor<4x64x128x3xbf16>) -> tensor<4x64x128x3xbf16>
    return %1 : tensor<4x64x128x3xbf16>
  }

  func.func @upsample2d_scale_nonunifrom(%arg0: tensor<4x32x64x3xbf16>) -> tensor<4x64x64x3xbf16> {
    %0 = ttir.empty() : tensor<4x64x64x3xbf16>
    %1 = "ttir.upsample2d"(%arg0, %0) <{scale_factor = array<i32: 2, 1>}> : (tensor<4x32x64x3xbf16>, tensor<4x64x64x3xbf16>) -> tensor<4x64x64x3xbf16>
    return %1 : tensor<4x64x64x3xbf16>
  }
}
