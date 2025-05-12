// REQUIRES: stablehlo
// RUN: not ttmlir-opt --automatic-sharding-pipeline="mesh-shape=1,2" %s 2>&1 | FileCheck %s

func.func public @gspmd_negative(%arg0: tensor<8192x800xf32>) -> (tensor<8192x800xf32> {jax.result_info = ""}) {
  %0 = stablehlo.custom_call @Sharding(%arg0) {backend_config = "", mhlo.sharding = "{devices=[1,2]<=[2]}"} : (tensor<8192x800xf32>) -> tensor<8192x800xf32>
  %1 = stablehlo.custom_call @SPMDFullToShardShape(%0) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x800xf32>) -> tensor<8192x400xf32>
  %2 = "stablehlo.all_gather"(%1) <{all_gather_dim = 1 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0], [1]]> : tensor<2x1xi64>, use_global_device_ids}> : (tensor<8192x400xf32>) -> tensor<8192x400xf32>
  %3 = stablehlo.custom_call @Sharding(%2) {backend_config = "", mhlo.sharding = "{manual}"} : (tensor<8192x400xf32>) -> tensor<8192x400xf32>
  %4 = stablehlo.custom_call @SPMDShardToFullShape(%3) {backend_config = "", mhlo.sharding = "{devices=[1,2]<=[2]}"} : (tensor<8192x400xf32>) -> tensor<8192x800xf32>
  return %4 : tensor<8192x800xf32>
}

// CHECK: error: Shardy automatic parallelization pass does not support GSPMD annotated module
