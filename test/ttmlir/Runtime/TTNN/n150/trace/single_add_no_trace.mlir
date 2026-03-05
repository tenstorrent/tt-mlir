// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

module {
  func.func @single_add(%arg0: tensor<256x256xbf16>, %arg1: tensor<256x256xbf16>) -> tensor<256x256xbf16> {
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<256x256xbf16>, tensor<256x256xbf16>) -> tensor<256x256xbf16>
    return %0 : tensor<256x256xbf16>
  }
}
