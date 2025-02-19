// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" --tt-unwrap-device-module %s > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %basename_t.ttnn
// RUN: ttmlir-opt --ttnn-modify-signatures-for-dylib --convert-ttnn-to-emitc %t.mlir > %t2.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir > %basename_t.cpp

func.func @reshape(%arg0: tensor<4x2x32x32xbf16>) -> tensor<2x4x32x32xbf16> {
  %0 = tensor.empty() : tensor<2x4x32x32xbf16>
  %1 = "ttir.reshape"(%arg0, %0) <{shape = [2: i32, 4: i32, 32: i32, 32: i32]}> : (tensor<4x2x32x32xbf16>, tensor<2x4x32x32xbf16>) -> tensor<2x4x32x32xbf16>
  return %1 : tensor<2x4x32x32xbf16>
}
