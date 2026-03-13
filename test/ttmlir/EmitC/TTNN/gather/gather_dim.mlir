// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t.mlir
// RUN: ttmlir-opt --ttnn-to-emitc-device-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

module {
  func.func @forward(%arg0: tensor<32x64xbf16>, %arg1: tensor<32x16xui32>) -> tensor<32x16xbf16> {
    %1 = "ttir.gather_dim"(%arg0, %arg1) <{dim = 1 : i32}> : (tensor<32x64xbf16>, tensor<32x16xui32>) -> tensor<32x16xbf16>
    return %1 : tensor<32x16xbf16>
  }
}
