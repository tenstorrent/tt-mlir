// RUN: ttmlir-opt --ttir-to-ttmetal-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttmetal-to-flatbuffer %t.mlir > %t.ttm

func.func @matmul(%arg0: tensor<64x128xf32>, %arg1: tensor<128x64xf32>) -> tensor<64x64xf32> {
  // CHECK: %[[C:.*]] = "ttmetal.create_buffer"[[C:.*]]
  %0 = tensor.empty() : tensor<64x64xf32>
  // CHECK: %[[C:.*]] = "ttmetal.enqueue_program"[[C:.*]]
  %1 = "ttir.matmul"(%arg0, %arg1, %0) : (tensor<64x128xf32>, tensor<128x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
  return %1 : tensor<64x64xf32>
}
