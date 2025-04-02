// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
module @jit_add attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<64x64xf32> {mhlo.sharding = "{replicated}"}, %arg1: tensor<64x64xf32> {mhlo.sharding = "{replicated}"}) -> (tensor<64x64xf32> {jax.result_info = ""}) {
    %0 = stablehlo.add %arg0, %arg1 : tensor<64x64xf32>
    %1 = interpreter.probe %0, probe_id = "probe1" : tensor<64x64xf32>
    return %1 : tensor<64x64xf32>
  }
}