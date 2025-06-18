// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --convert-ttir-to-ttnn %s > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %basename_t.ttnn
// RUN: ttmlir-opt --ttnn-tuplify-tensors --convert-ttnn-to-emitc %t.mlir > %t2.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir > %basename_t.cpp

// UNSUPPORTED: true
// Marked as UNSUPPORTED because it causes segfault on subsequent test.
// https://github.com/tenstorrent/tt-mlir/issues/3511

#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x32xf32, #system_memory>>

module {
  func.func @empty_host() -> tensor<32x32xf32, #ttnn_layout> {
    %0 = ttir.empty() : tensor<32x32xf32, #ttnn_layout>
    return %0 : tensor<32x32xf32, #ttnn_layout>
  }
}
