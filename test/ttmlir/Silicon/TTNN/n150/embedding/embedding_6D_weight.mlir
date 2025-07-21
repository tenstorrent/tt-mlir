// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

func.func @embedding_weight_6D(%input: tensor<2x4xui32>, %weight: tensor<1x1x1x1x10x10xbf16>) -> tensor<2x4x10xbf16> {
  // CHECK-LABEL: func.func @embedding_weight_6D
  %output = ttir.empty() : tensor<2x4x10xbf16>
  // CHECK: "ttnn.reshape"
  // CHECK-SAME: shape = [1 : i32, 1 : i32, 10 : i32, 10 : i32]
  // CHECK: "ttnn.embedding"
  %result = "ttir.embedding"(%input, %weight, %output) : (tensor<2x4xui32>, tensor<1x1x1x1x10x10xbf16>, tensor<2x4x10xbf16>) -> tensor<2x4x10xbf16>
  return %result : tensor<2x4x10xbf16>
}
