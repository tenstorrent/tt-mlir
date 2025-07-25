// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %basename_t.ttnn
// RUN: ttmlir-opt --ttnn-tuplify-tensors --convert-ttnn-to-emitc %t.mlir > %t2.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir > %basename_t.cpp

func.func @forward(%arg0: tensor<128x10x32x4xbf16>) -> tensor<128x1x32x4xbf16> {
  %0 = ttir.empty() : tensor<128x1x32x4xbf16>
  %1 = "ttir.prod"(%arg0, %0) <{dim_arg = [1 : i32], keep_dim = true}> : (tensor<128x10x32x4xbf16>, tensor<128x1x32x4xbf16>) -> tensor<128x1x32x4xbf16>
  return %1 : tensor<128x1x32x4xbf16>
}
