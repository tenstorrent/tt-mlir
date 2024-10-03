// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
module @jit_module_reshape attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1x64x64x64xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) -> (tensor<1x1x4096x64xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.reshape %arg0 : (tensor<1x64x64x64xf32>) -> tensor<1x1x4096x64xf32>
    // CHECK: %[[C:.*]] = tensor.empty[[C:.*]]
    // CHECK: %[[C:.*]] = "ttir.reshape"[[C:.*]]
    return %0 : tensor<1x1x4096x64xf32>
  }
}
