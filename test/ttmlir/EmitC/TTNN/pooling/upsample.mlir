// TODO(dmilinkovic): re-enable CPU-hoisted const-eval once EmitC support for CPU-hoisted ops lands - issue #6100.
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-cpu-hoisted-const-eval=false system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t.mlir
// RUN: ttmlir-opt --ttnn-to-emitc-device-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

module {
  func.func @upsample2d_scale_unifrom(%arg0: tensor<4x32x64x3xbf16>) -> tensor<4x64x128x3xbf16> {
    %0 = "ttir.upsample2d"(%arg0) <{scale_factor = 2 : si32}> : (tensor<4x32x64x3xbf16>) -> tensor<4x64x128x3xbf16>
    return %0 : tensor<4x64x128x3xbf16>
  }

  func.func @upsample2d_scale_nonunifrom(%arg0: tensor<4x32x64x3xbf16>) -> tensor<4x64x64x3xbf16> {
    %0 = "ttir.upsample2d"(%arg0) <{scale_factor = array<i32: 2, 1>}> : (tensor<4x32x64x3xbf16>) -> tensor<4x64x64x3xbf16>
    return %0 : tensor<4x64x64x3xbf16>
  }
}
