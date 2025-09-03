// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
module @rms_norm {
  func.func public @test_rms_norm(%arg0: tensor<2x4xf32>, %arg1: tensor<4xf32>) -> tensor<2x4xf32> {
    %0 = ttir.empty() : tensor<2x4xf32>
    // CHECK: ttnn.rms_norm
    %1 = "ttir.rms_norm"(%arg0, %arg1, %0) <{normalized_shape = array<i64: 4>, epsilon = 1.000000e-05 : f32, operandSegmentSizes = array<i32: 1, 1, 0, 1>}> : (tensor<2x4xf32>, tensor<4xf32>, tensor<2x4xf32>) -> tensor<2x4xf32>
    return %1 : tensor<2x4xf32>
  }
}
