// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module @jit_module_relu attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<32x32xf32> {mhlo.layout_mode = "default", mhlo.sharding = "{replicated}"}) -> (tensor<32x32xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<32x32xf32>
    %1 = stablehlo.maximum %arg0, %0 : tensor<32x32xf32>
    return %1 : tensor<32x32xf32>
  }
}
