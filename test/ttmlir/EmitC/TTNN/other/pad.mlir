// TODO(dmilinkovic): re-enable CPU-hoisted const-eval once EmitC support for CPU-hoisted ops lands - issue #6100.
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-cpu-hoisted-const-eval=false system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t.mlir
// RUN: ttmlir-opt --ttnn-to-emitc-device-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir
//
// UNSUPPORTED: true
// Outstanding bug: https://github.com/tenstorrent/tt-mlir/issues/2072
module {
  func.func @pad(%arg0: tensor<1x1x5x5xbf16>) -> tensor<1x1x7x7xbf16> {
    // CHECK: ttnn.pad
    %1 = "ttir.pad"(%arg0) <{padding = array<i32: 0, 0, 0, 0, 1, 1, 1, 1>, value = 0.000000e+00 : f32}> : (tensor<1x1x5x5xbf16>) -> tensor<1x1x7x7xbf16>
    return %1 : tensor<1x1x7x7xbf16>
  }
}
