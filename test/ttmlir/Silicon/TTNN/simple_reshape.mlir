// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

func.func @forward(%arg0: tensor<4x2x32x32xbf16>) -> tensor<2x4x32x32xbf16> {
  %0 = ttir.empty() : tensor<2x4x32x32xbf16>
  // CHECK: %[[C:.*]] = "ttnn.reshape"[[C:.*]]
  %1 = "ttir.reshape"(%arg0, %0) <{shape = [2: i32, 4: i32, 32: i32, 32: i32]}> : (tensor<4x2x32x32xbf16>, tensor<2x4x32x32xbf16>) -> tensor<2x4x32x32xbf16>
  return %1 : tensor<2x4x32x32xbf16>
}
