// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t.mlir
// RUN: ttmlir-opt --ttnn-backend-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

func.func @forward(%arg0: tensor<32x32x64xbf16>) -> tensor<1x1x1xbf16> {
  %0 = ttir.empty() : tensor<1x1x1xbf16>
  %1 = "ttir.max"(%arg0, %0) <{keep_dim = true}> : (tensor<32x32x64xbf16>, tensor<1x1x1xbf16>) -> tensor<1x1x1xbf16>
  return %1 : tensor<1x1x1xbf16>
}
