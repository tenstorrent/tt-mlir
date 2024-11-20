// RUN: ttmlir-opt --ttir-reshape-fold %s| FileCheck %s
#any_device_tile = #tt.operand_constraint<dram|l1|tile|none|interleaved|single_bank|height_sharded|width_sharded|block_sharded|any_layout|any_device_tile>
module @jit_ravel attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<1xi32>) -> (tensor<1xi32> {jax.result_info = ""}) {
    %0 = tensor.empty() : tensor<1xi32>
    %1 = "ttir.reshape"(%arg0, %0) <{operand_constraints = [#any_device_tile, #any_device_tile], shape = [1 : i32]}> : (tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>
    // CHECK: return %arg0 : tensor<1xi32>
    return %1 : tensor<1xi32>
  }
}

