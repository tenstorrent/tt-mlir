// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

func.func @round(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // CHECK: %{{[0-9]+}} = "ttnn.round"
  %1 = "ttir.round"(%arg0) : (tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}
