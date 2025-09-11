// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t.mlir
// RUN: ttmlir-opt --ttnn-backend-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

func.func @cumsum(%arg0: tensor<4x4x128x128xbf16>) -> tensor<4x4x128x128xbf16> {
  %0 = ttir.empty() : tensor<4x4x128x128xbf16>
  %1 = "ttir.cumsum"(%arg0, %0) <{dim = 1 : i64}> : (tensor<4x4x128x128xbf16>, tensor<4x4x128x128xbf16>) -> tensor<4x4x128x128xbf16>
  return %1 : tensor<4x4x128x128xbf16>
}
