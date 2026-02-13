// TODO(dmilinkovic): re-enable CPU-hoisted const-eval once EmitC support for CPU-hoisted ops lands - issue #6100.
// RUN: ttmlir-opt --ttir-to-ttnn-common-pipeline="enable-cpu-hoisted-const-eval=false system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-opt --ttnn-common-to-flatbuffer-pipeline -o %t_fb.mlir %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t_fb.mlir
// RUN: ttmlir-opt --ttnn-common-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

module {
    func.func @gelu_bw_default(%arg0: tensor<4x4xbf16>, %arg1: tensor<4x4xbf16>) -> tensor<4x4xbf16> {
        %1 = "ttir.gelu_bw"(%arg0, %arg1) : (tensor<4x4xbf16>, tensor<4x4xbf16>) -> tensor<4x4xbf16>
      return %1 : tensor<4x4xbf16>
    }
    func.func @gelu_bw(%arg0: tensor<4x4xbf16>, %arg1: tensor<4x4xbf16>) -> tensor<4x4xbf16> {
        %1 = "ttir.gelu_bw"(%arg0, %arg1) <{approximate = "tanh"}> : (tensor<4x4xbf16>, tensor<4x4xbf16>) -> tensor<4x4xbf16>
      return %1 : tensor<4x4xbf16>
    }
}
