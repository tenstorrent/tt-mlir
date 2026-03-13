// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="mesh-shape=1,2 system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t.mlir
// RUN: ttmlir-opt --ttnn-to-emitc-device-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

module {
  func.func @forward(%arg0: tensor<128x128xbf16>) -> tensor<128x128xbf16> {
    %1 = "ttir.all_reduce_async"(%arg0) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<128x128xbf16>) -> tensor<128x128xbf16>
    return %1 : tensor<128x128xbf16>
  }
}
