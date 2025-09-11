// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t.mlir
// RUN: ttmlir-opt --ttnn-backend-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

func.func @forward(%arg0: tensor<512x1024xbf16>) -> tensor<512x1xbf16> {
  %0 = ttir.empty() : tensor<512x1xbf16>
  %1 = "ttir.mean"(%arg0, %0) <{dim_arg = [-1: i32], keep_dim = true}> : (tensor<512x1024xbf16>, tensor<512x1xbf16>) -> tensor<512x1xbf16>
  return %1 : tensor<512x1xbf16>
}
