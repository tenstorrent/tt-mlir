// REQUIRES: stablehlo, num-chips-1 || num-chips-2
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | \
// RUN:     ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
// RUN: FileCheck --input-file=%t.mlir %s

module @jit_dot_general_4d attributes {} {
  func.func public @test_dot_general_4d(%arg0 : tensor<1x128x16x32xf32>, %arg1 : tensor<1x128x32x8xf32>) -> tensor<1x128x16x8xf32> {
    // CHECK-LABEL: func.func public @test_dot_general
    // CHECK: ttnn.empty
    // CHECK: ttnn.matmul
    // CHECK-SAME: tensor<1x128x16x32xf32,
    // CHECK-SAME: tensor<1x128x32x8xf32,
    // CHECK-SAME: tensor<1x128x16x8xf32,
    // CHECK-SAME: -> tensor<1x128x16x8xf32
    %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0, 1] x [0, 1], contracting_dims = [3] x [2] : (tensor<1x128x16x32xf32>, tensor<1x128x32x8xf32>) -> tensor<1x128x16x8xf32>
    return %0 : tensor<1x128x16x8xf32>
  }
}
