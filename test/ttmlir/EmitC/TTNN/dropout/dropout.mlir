// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t.mlir
// RUN: ttmlir-opt --ttnn-to-emitc-device-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

module {
    func.func @dropout(%arg0: tensor<64x128xbf16>) -> tensor<64x128xbf16> {
        %1 = "ttir.dropout"(%arg0) <{prob = 0.2 : f32, scale = 1.25 : f32, seed = 21 : ui32, use_per_device_seed = true}> : (tensor<64x128xbf16>) -> tensor<64x128xbf16>
      return %1 : tensor<64x128xbf16>
    }
}
