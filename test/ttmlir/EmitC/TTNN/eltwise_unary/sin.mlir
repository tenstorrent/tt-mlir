// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %basename_t.ttnn
// RUN: ttmlir-opt --ttnn-modify-signatures-for-dylib --convert-ttnn-to-emitc %t.mlir > %t2.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir > %basename_t.cpp

func.func @sin(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %0 = tensor.empty() : tensor<32x32xf32>
  %1 = "ttir.sin"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %1 : tensor<32x32xf32>
}
