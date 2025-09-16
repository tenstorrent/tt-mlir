// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t.mlir
// RUN: ttmlir-opt --ttnn-backend-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

func.func @repeat_interleave(%arg0: tensor<4x6xf32>) -> tensor<4x24xf32> {
  %0 = ttir.empty() : tensor<4x24xf32>
  %1 = "ttir.repeat_interleave"(%arg0, %0) {repeats = 4 : ui32, dim = 1 : si32} : (tensor<4x6xf32>, tensor<4x24xf32>) -> tensor<4x24xf32>
  return %1 : tensor<4x24xf32>
}
