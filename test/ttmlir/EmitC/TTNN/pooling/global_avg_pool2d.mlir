// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t.mlir
// RUN: ttmlir-opt --ttnn-backend-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

module attributes {} {
  func.func @global_avg_pool2d(%arg0: tensor<1x128x128x32xbf16>) -> tensor<1x1x1x32xbf16> {
    %0 = ttir.empty() : tensor<1x1x1x32xbf16>
    %1 = "ttir.global_avg_pool2d"(%arg0, %0) : (tensor<1x128x128x32xbf16>, tensor<1x1x1x32xbf16>) -> tensor<1x1x1x32xbf16>
    return %1 : tensor<1x1x1x32xbf16>
  }
}
