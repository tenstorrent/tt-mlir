// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
module @jit_broadcast attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1xf32> {mhlo.layout_mode = "default"}, %arg1: tensor<512x512xf32> {mhlo.layout_mode = "default"}) -> (tensor<512x512xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<1xf32>) -> tensor<512x512xf32>
    %1 = stablehlo.maximum %arg1, %0 : tensor<512x512xf32>
    // CHECK: %[[C:.*]] = "ttir.broadcast"[[C:.*]]
    return %1 : tensor<512x512xf32>
  }
}
