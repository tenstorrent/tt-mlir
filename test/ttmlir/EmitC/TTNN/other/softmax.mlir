// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %basename_t.ttnn
// RUN: ttmlir-opt --ttnn-tuplify-tensors --convert-ttnn-to-emitc %t.mlir > %t2.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir > %basename_t.cpp

func.func @softmax(%arg0: tensor<512x1024xbf16>) -> tensor<512x1024xbf16> {
  %0 = ttir.empty() : tensor<512x1024xbf16>
  %1 = "ttir.softmax"(%arg0, %0) <{dimension = 1 : si32}> : (tensor<512x1024xbf16>, tensor<512x1024xbf16>) -> tensor<512x1024xbf16>
  return %1 : tensor<512x1024xbf16>
}
