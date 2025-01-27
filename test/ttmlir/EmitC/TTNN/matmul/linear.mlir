// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %basename_t.ttnn
// RUN: ttmlir-opt --ttnn-modify-signatures-for-dylib --convert-ttnn-to-emitc %t.mlir > %t2.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir > %basename_t.cpp

func.func @linear_with_bias(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x64xbf16>, %bias: tensor<64x64xbf16>) -> tensor<64x64xbf16> {
  %0 = tensor.empty() : tensor<64x64xbf16>
  %1 = "ttir.linear"(%arg0, %arg1, %bias, %0) : (tensor<64x128xbf16>, tensor<128x64xbf16>, tensor<64x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
  return %1 : tensor<64x64xbf16>
}

func.func @linear(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x64xbf16>) -> tensor<64x64xbf16> {
  %0 = tensor.empty() : tensor<64x64xbf16>
  %1 = "ttir.linear"(%arg0, %arg1, %0) : (tensor<64x128xbf16>, tensor<128x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
  return %1 : tensor<64x64xbf16>
}