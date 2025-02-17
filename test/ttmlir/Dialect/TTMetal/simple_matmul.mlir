// RUN: ttmlir-opt --ttir-to-ttmetal-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttmetal-to-flatbuffer %t.mlir > %t.ttm

func.func @matmul(%arg0: tensor<1x1x2x4x!tt.tile<32x32, f32>>, %arg1: tensor<1x1x4x2x!tt.tile<32x32, f32>>) -> tensor<1x1x2x2x!tt.tile<32x32, f32>> {
  // CHECK: %[[C:.*]] = "ttmetal.create_buffer"[[C:.*]]
  %0 = tensor.empty() : tensor<1x1x2x2x!tt.tile<32x32, f32>>
  // CHECK: %[[C:.*]] = "ttmetal.enqueue_program"[[C:.*]]
  %1 = "ttir.matmul"(%arg0, %arg1, %0) : (tensor<1x1x2x4x!tt.tile<32x32, f32>>, tensor<1x1x4x2x!tt.tile<32x32, f32>>, tensor<1x1x2x2x!tt.tile<32x32, f32>>) -> tensor<1x1x2x2x!tt.tile<32x32, f32>>
  return %1 : tensor<1x1x2x2x!tt.tile<32x32, f32>>
}


