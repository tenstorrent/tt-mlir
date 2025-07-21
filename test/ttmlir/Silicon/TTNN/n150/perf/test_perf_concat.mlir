// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
func.func @concat(%arg0: tensor<32x32xf32>, %arg1: tensor<32x64xf32>) -> tensor<32x96xf32> {
  %0 = ttir.empty() : tensor<32x96xf32>
  // CHECK: = "ttnn.concat"
  %1 = "ttir.concat"(%arg0, %arg1, %0) <{dim = 1 : si32}> : (tensor<32x32xf32>, tensor<32x64xf32>, tensor<32x96xf32>) -> tensor<32x96xf32>
  return %1 : tensor<32x96xf32>
}
