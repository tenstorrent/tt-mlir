// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

func.func @squeeze(%arg0: tensor<1x2x1x32x32xbf16>) -> tensor<1x2x32x32xbf16> {
  %0 = ttir.empty() : tensor<1x2x32x32xbf16>
  // CHECK: = "ttnn.reshape"
  %1 = "ttir.squeeze"(%arg0, %0) <{dim = 2 : si32}> : (tensor<1x2x1x32x32xbf16>, tensor<1x2x32x32xbf16>) -> tensor<1x2x32x32xbf16>
  return %1 : tensor<1x2x32x32xbf16>
}
