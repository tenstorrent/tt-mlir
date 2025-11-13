// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t.mlir
// RUN: ttmlir-opt --ttnn-backend-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

module {
    func.func @gelu_bw_default(%arg0: tensor<4x4xbf16>, %arg1: tensor<4x4xbf16>) -> tensor<4x4xbf16> {
        %0 = ttir.empty() : tensor<4x4xbf16>
        %1 = "ttir.gelu_bw"(%arg0, %arg1, %0) : (tensor<4x4xbf16>, tensor<4x4xbf16>, tensor<4x4xbf16>) -> tensor<4x4xbf16>
      return %1 : tensor<4x4xbf16>
    }
    func.func @gelu_bw(%arg0: tensor<4x4xbf16>, %arg1: tensor<4x4xbf16>) -> tensor<4x4xbf16> {
        %0 = ttir.empty() : tensor<4x4xbf16>
        %1 = "ttir.gelu_bw"(%arg0, %arg1, %0) <{approximate = "tanh"}> : (tensor<4x4xbf16>, tensor<4x4xbf16>, tensor<4x4xbf16>) -> tensor<4x4xbf16>
      return %1 : tensor<4x4xbf16>
    }
}
