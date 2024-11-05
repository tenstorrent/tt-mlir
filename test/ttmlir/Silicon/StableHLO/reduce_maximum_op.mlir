// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | \
// RUN:     ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
// RUN: FileCheck --input-file=%t.mlir %s
// UNSUPPORTED: true
// error: keepdim=False is not supported

module @jit_reduce_maximum attributes {} {
  func.func public @test_reduce_maximum(%arg0: tensor<128x10xf32>, %cst_0: tensor<f32>) -> tensor<128xf32> {
    // CHECK-LABEL: func.func public @test_reduce_maximum
    // CHECK: ttnn.max
    // CHECK-SAME: dim_arg = [1 : i32],
    // CHECK-SAME: keep_dim = false}
    // CHECK-SAME: tensor<128x10xf32,
    // CHECK-SAME: -> tensor<128xf32
    %0 = stablehlo.reduce(%arg0 init: %cst_0) applies stablehlo.maximum across dimensions = [1] : (tensor<128x10xf32>, tensor<f32>) -> tensor<128xf32>
    return %0 : tensor<128xf32>
  }
}
