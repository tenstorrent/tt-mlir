// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
func.func @sum(%arg0: tensor<1x1x512x64xbf16>) -> tensor<1x1x512x1xbf16> {
  %0 = tensor.empty() : tensor<1x1x512x1xbf16>
  // CHECK: = "ttnn.sum"
  %1 = "ttir.sum"(%arg0, %0) <{dim_arg = [-1: i32], keep_dim = true}> : (tensor<1x1x512x64xbf16>, tensor<1x1x512x1xbf16>) -> tensor<1x1x512x1xbf16>
  return %1 : tensor<1x1x512x1xbf16>
}
