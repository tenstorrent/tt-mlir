// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-repeat-folding-workaround-pass=false" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t.mlir
// RUN: ttmlir-opt --ttnn-backend-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

// UNSUPPORTED: true
// Marked as UNSUPPORTED because of segfault: https://github.com/tenstorrent/tt-mlir/issues/3266

func.func @repeat(%arg0: tensor<1x32x32xf32>) -> tensor<32x32x32xf32> {
  %0 = ttir.empty() : tensor<32x32x32xf32>
  %1 = "ttir.repeat"(%arg0, %0) {repeat_dimensions = array<i64: 32, 1, 1>} : (tensor<1x32x32xf32>, tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
  return %1 : tensor<32x32x32xf32>
}
