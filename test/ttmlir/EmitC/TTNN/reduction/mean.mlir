// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %basename_t.ttnn
// RUN: ttmlir-opt --ttnn-modify-signatures-for-dylib --convert-ttnn-to-emitc %t.mlir > %t2.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir > %basename_t.cpp
//
// UNSUPPORTED: true
// This test works in EmitC path, but not in TTRT path:
// https://github.com/tenstorrent/tt-mlir/issues/1942

func.func @forward(%arg0: tensor<512x1024xbf16>) -> tensor<512x32xbf16> {
  %0 = tensor.empty() : tensor<512x32xbf16>
  %1 = "ttir.mean"(%arg0, %0) <{dim_arg = [-1: i32], keep_dim = true}> : (tensor<512x1024xbf16>, tensor<512x32xbf16>) -> tensor<512x32xbf16>
  return %1 : tensor<512x32xbf16>
}
