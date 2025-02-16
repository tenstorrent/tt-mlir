// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %basename_t.ttnn
// RUN: ttmlir-opt --ttnn-modify-signatures-for-dylib --convert-ttnn-to-emitc %t.mlir > %t2.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir > %basename_t.cpp

func.func @repeat_interleave(%arg0: tensor<4x6xf32>) -> tensor<4x24xf32> {
  %0 = tensor.empty() : tensor<4x24xf32>
  %1 = "ttir.repeat_interleave"(%arg0, %0) {repeats = 4 : ui32, dim = 1 : si32} : (tensor<4x6xf32>, tensor<4x24xf32>) -> tensor<4x24xf32>
  return %1 : tensor<4x24xf32>
}
