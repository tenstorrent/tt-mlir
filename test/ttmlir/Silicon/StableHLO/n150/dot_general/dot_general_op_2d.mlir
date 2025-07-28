// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

module @jit_dot_general_2d attributes {} {
  func.func public @test_dot_general_2d(%arg0 : tensor<16x32xf32>, %arg1 : tensor<32x8xf32>) -> tensor<16x8xf32> {
    // CHECK-LABEL: func.func public @test_dot_general
    // CHECK: ttnn.matmul
    // CHECK-SAME: tensor<16x32xf32,
    // CHECK-SAME: tensor<32x8xf32,
    // CHECK-SAME: -> tensor<16x8xf32
    %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<16x32xf32>, tensor<32x8xf32>) -> tensor<16x8xf32>
    return %0 : tensor<16x8xf32>
  }
}
