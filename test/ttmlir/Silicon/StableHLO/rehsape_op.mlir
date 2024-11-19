// REQUIRES: stablehlo
// RUN: rm -rf %t.ttnn
// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | \
// RUN:     ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
// RUN: FileCheck --input-file=%t.mlir %s

module @jit_module_reshape attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @test_reshape(%arg0: tensor<1x64x64x64xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) -> (tensor<1x1x4096x64xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    // CHECK-LABEL: func.func public @test_reshape
    // CHECK: ttnn.reshape
    // CHECK-SAME: {shape = [1 : i32, 1 : i32, 4096 : i32, 64 : i32]}
    // CHECK-SAME: tensor<1x64x64x64xf32
    // CHECK-SAME: -> tensor<1x1x4096x64xf32
    %0 = stablehlo.reshape %arg0 : (tensor<1x64x64x64xf32>) -> tensor<1x1x4096x64xf32>
    return %0 : tensor<1x1x4096x64xf32>
  }
}
