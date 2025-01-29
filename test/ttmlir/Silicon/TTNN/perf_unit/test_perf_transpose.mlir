// REQUIRES: num-chips-1 || num-chips-2
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
func.func @transpose(%arg0: tensor<64x128xbf16>) -> tensor<128x64xbf16> {
  %0 = tensor.empty() : tensor<128x64xbf16>
  // CHECK: %[[C:.*]] = "ttnn.transpose"[[C:.*]]
  %1 = "ttir.transpose"(%arg0, %0) <{dim0 = 0 : si32, dim1 = 1 : si32}> : (tensor<64x128xbf16>, tensor<128x64xbf16>) -> tensor<128x64xbf16>
  return %1 : tensor<128x64xbf16>
}
