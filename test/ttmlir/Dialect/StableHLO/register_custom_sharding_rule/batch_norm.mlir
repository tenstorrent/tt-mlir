// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-pipeline -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

// Test that batch_norm_training sharding rule correctly partitions the feature
// dimension across all operands and results. The input tensor<2x6xbf16> is
// sharded on dim 0 (batch) with _axis_0=2, then reshaped to tensor<1x12x1xbf16>
// which merges batch into the feature dim. The sharding rule should propagate
// _axis_0 through the feature dimension so that scale, bias, mean, and variance
// (all 1-D) are partitioned consistently: 12 -> 6.
module @batch_norm_training_feature_sharding attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  sdy.mesh @mesh = <["_axis_0"=2]>
  func.func @main(
    %arg0: tensor<2x6xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{"_axis_0"}, {}]>, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "input"},
    %arg1: tensor<12xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "scale"},
    %arg2: tensor<12xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "bias"}
  ) -> tensor<2x6xbf16> {
    %0 = stablehlo.reshape %arg0 : (tensor<2x6xbf16>) -> tensor<1x12x1xbf16>
    %output, %mean, %var = "stablehlo.batch_norm_training"(%0, %arg1, %arg2) <{epsilon = 1.0e-5 : f32, feature_index = 1 : i64}> : (tensor<1x12x1xbf16>, tensor<12xbf16>, tensor<12xbf16>) -> (tensor<1x12x1xbf16>, tensor<12xbf16>, tensor<12xbf16>)
    %1 = stablehlo.reshape %output : (tensor<1x12x1xbf16>) -> tensor<2x6xbf16>
    return %1 : tensor<2x6xbf16>
  }
}

// (a) N-D output feature dim is partitioned: 1x12x1 -> 1x6x1
// (b) 1-D mean/variance are partitioned the same way: 12 -> 6
// CHECK: "stablehlo.batch_norm_training"
// CHECK-SAME: (tensor<1x6x1xbf16>, tensor<6xbf16>, tensor<6xbf16>)
// CHECK-SAME: -> (tensor<1x6x1xbf16>, tensor<6xbf16>, tensor<6xbf16>)
