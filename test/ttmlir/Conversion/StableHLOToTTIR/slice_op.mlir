// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
module @jit_module_slice attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<64x64xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) -> (tensor<32x32xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.slice %arg0 [0:32, 0:32] : (tensor<64x64xf32>) -> tensor<32x32xf32>
    return %0 : tensor<32x32xf32>
  }
}