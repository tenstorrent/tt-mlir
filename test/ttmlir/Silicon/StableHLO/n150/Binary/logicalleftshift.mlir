// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

module @jit_eltwise_logicalshiftleft attributes {} {
  func.func public @test_logicalshiftleft(%arg0: tensor<5xui32>, %arg1: tensor<5xui32>) -> tensor<5xui32> {
    // CHECK-LABEL: func.func public @test_logicalshiftleft
    // CHECK: ttnn.logical_left_shift
    // CHECK-SAME: tensor<5xui32,
    // CHECK-SAME: tensor<5xui32,
    // CHECK-SAME: -> tensor<5xui32,
    %0 = stablehlo.shift_left %arg0, %arg1 : tensor<5xui32>
    return %0 : tensor<5xui32>
  }
}
