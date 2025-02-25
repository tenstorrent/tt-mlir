// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %basename_t.ttnn
// RUN: ttmlir-opt --ttnn-modify-signatures-for-dylib --convert-ttnn-to-emitc %t.mlir > %t2.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir > %basename_t.cpp

func.func @cumsum(%arg0: tensor<4x4x128x128xbf16>) -> tensor<4x4x128x128xbf16> {
  %0 = tensor.empty() : tensor<4x4x128x128xbf16>
  %1 = "ttir.cumsum"(%arg0, %0) <{dim = 1 : i64}> : (tensor<4x4x128x128xbf16>, tensor<4x4x128x128xbf16>) -> tensor<4x4x128x128xbf16>
  return %1 : tensor<4x4x128x128xbf16>
}
