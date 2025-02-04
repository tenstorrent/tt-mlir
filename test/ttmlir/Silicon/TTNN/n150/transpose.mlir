// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
func.func @transpose(%arg0: tensor<64x128xbf16>) -> tensor<128x64xbf16> {
  %0 = tensor.empty() : tensor<128x64xbf16>
  // CHECK: %[[C:.*]] = "ttnn.transpose"[[C:.*]]
  %1 = "ttir.transpose"(%arg0, %0) <{dim0 = 0 : si32, dim1 = 1 : si32}> : (tensor<64x128xbf16>, tensor<128x64xbf16>) -> tensor<128x64xbf16>
  return %1 : tensor<128x64xbf16>
}

func.func @transpose_8x8(%arg0: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
  %0 = tensor.empty() : tensor<32x32xbf16>
  // CHECK: %[[C:.*]] = "ttnn.transpose"[[C:.*]]
  %1 = "ttir.transpose"(%arg0, %0) <{dim0 = 0 : si32, dim1 = 1 : si32}> : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  return %1 : tensor<32x32xbf16>
}

func.func @transpose_8x16_reverse_dims(%arg0: tensor<64x16xbf16>) -> tensor<16x64xbf16> {
  %0 = tensor.empty() : tensor<16x64xbf16>
  // CHECK: %[[C:.*]] = "ttnn.transpose"[[C:.*]]
  %1 = "ttir.transpose"(%arg0, %0) <{dim0 = 1 : si32, dim1 = 0 : si32}> : (tensor<64x16xbf16>, tensor<16x64xbf16>) -> tensor<16x64xbf16>
  return %1 : tensor<16x64xbf16>
}

func.func @transpose_negative_dims(%arg0: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
  %0 = tensor.empty() : tensor<32x32xbf16>
  // CHECK: %[[C:.*]] = "ttnn.transpose"[[C:.*]]
  %1 = "ttir.transpose"(%arg0, %0) <{dim0 = -1 : si32, dim1 = -2 : si32}> : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  return %1 : tensor<32x32xbf16>
}
