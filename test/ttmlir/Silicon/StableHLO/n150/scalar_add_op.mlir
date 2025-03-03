// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | \
// RUN:     ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
// RUN: FileCheck --input-file=%t.mlir %s
module @jit_eltwise_scalar_add attributes {} {
  func.func public @test_scalar_add(%arg0: tensor<f32>, %arg1: tensor<f32>) -> tensor<f32> {
    // CHECK-LABEL: func.func public @test_scalar_add
    // CHECK: ttnn.add
    // CHECK-SAME: tensor<1xf32,
    // CHECK-SAME: tensor<1xf32,
    // CHECK-SAME: -> tensor<1xf32,
    %0 = stablehlo.add %arg0, %arg1 : tensor<f32>
    return %0 : tensor<f32>
  }
}
