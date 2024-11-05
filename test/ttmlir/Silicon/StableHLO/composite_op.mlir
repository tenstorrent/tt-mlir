// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | \
// RUN:     ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
// RUN: FileCheck --input-file=%t.mlir %s

module @jit_eltwise_add attributes {} {
  // CHECK-NOT: func.func.private @add_impl
  func.func private @add_impl(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<64x128xf32>
    return %0 : tensor<64x128xf32>
  }

  // CHECK-LABEL: func.func public @main
  func.func public @main(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    // CHECK: ttnn.empty
    // CEHCK: ttnn.add
    %results = stablehlo.composite "jit_eltwise_add.my_add" %arg0, %arg1 {
        decomposition = @add_impl
    } : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    return %results : tensor<64x128xf32>
  }
}
