// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %basename_t.ttnn
// RUN: ttmlir-opt --ttnn-modify-signatures-for-dylib --convert-ttnn-to-emitc %t.mlir > %t2.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir > %basename_t.cpp
//
// UNSUPPORTED: true
// Outstanding bug: https://github.com/tenstorrent/tt-mlir/issues/2072
module {
  func.func @main(%arg0: tensor<1x1x5x5xbf16>) -> tensor<1x1x7x7xbf16> {
    // CHECK: ttnn.pad
    %1 = "ttir.pad"(%arg0) <{padding = array<i32: 0, 0, 0, 0, 1, 1, 1, 1>, value = 0.000000e+00 : f32}> : (tensor<1x1x5x5xbf16>) -> tensor<1x1x7x7xbf16>
    return %1 : tensor<1x1x7x7xbf16>
  }
}
