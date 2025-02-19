// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" --tt-unwrap-device-module %s > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %basename_t.ttnn
// RUN: ttmlir-opt --ttnn-modify-signatures-for-dylib --convert-ttnn-to-emitc %t.mlir > %t2.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir > %basename_t.cpp

func.func @repeat(%arg0: tensor<1x32x32xf32>) -> tensor<32x32x32xf32> {
  %0 = tensor.empty() : tensor<32x32x32xf32>
  %1 = "ttir.repeat"(%arg0, %0) {repeat_dimensions = array<i64: 32, 1, 1>} : (tensor<1x32x32xf32>, tensor<32x32x32xf32>) -> tensor<32x32x32xf32>
  return %1 : tensor<32x32x32xf32>
}
