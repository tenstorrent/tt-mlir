// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s
module @jit_eltwise_scalar_add attributes {} {
  func.func public @test_scalar_add(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    // CHECK-LABEL: func.func public @test_scalar_add
    // CHECK: ttnn.add
    // CHECK-SAME: tensor<f32,
    // CHECK-SAME: tensor<f32,
    // CHECK-SAME: -> tensor<f32,
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
}
