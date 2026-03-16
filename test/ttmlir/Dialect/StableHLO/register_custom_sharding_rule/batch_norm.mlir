// REQUIRES: stablehlo
// RUN: ttmlir-opt -split-input-file --stablehlo-pipeline -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

// Test 1: batch_norm_training, feature dim sharded (no data parallel).
// Scale is sharded on the feature dimension (kPassThrough), so sharding
// propagates without inserting any collective ops.
module @batch_norm_training_feature_sharding attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  sdy.mesh @mesh = <["model"=2]>
  func.func @main(
    %arg0: tensor<4x8xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "input"},
    %arg1: tensor<8xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{"model"}]>, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "scale"},
    %arg2: tensor<8xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "bias"}
  ) -> tensor<4x8xbf16> {
    %output, %mean, %var = "stablehlo.batch_norm_training"(%arg0, %arg1, %arg2) <{epsilon = 1.0e-5 : f32, feature_index = 1 : i64}> : (tensor<4x8xbf16>, tensor<8xbf16>, tensor<8xbf16>) -> (tensor<4x8xbf16>, tensor<8xbf16>, tensor<8xbf16>)
    return %output : tensor<4x8xbf16>
  }
}

// CHECK-NOT: stablehlo.all_gather
// CHECK-NOT: stablehlo.all_reduce
// CHECK: "stablehlo.batch_norm_training"
// CHECK-SAME: (tensor<4x4xbf16>, tensor<4xbf16>, tensor<4xbf16>) -> (tensor<4x4xbf16>, tensor<4xbf16>, tensor<4xbf16>)

// -----

// Test 2: batch_norm_training, batch dim + feature dim sharded (data parallel).
// Input is batch-sharded (non-feature dim, kNeedReplication), so Shardy inserts
// all_gather to replicate across the batch axis before the op.
module @batch_norm_training_data_parallel attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  sdy.mesh @mesh = <["model"=2, "batch"=4]>
  func.func @main(
    %arg0: tensor<4x8xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}]>, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "input"},
    %arg1: tensor<8xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{"model"}]>, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "scale"},
    %arg2: tensor<8xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "bias"}
  ) -> tensor<4x8xbf16> {
    %output, %mean, %var = "stablehlo.batch_norm_training"(%arg0, %arg1, %arg2) <{epsilon = 1.0e-5 : f32, feature_index = 1 : i64}> : (tensor<4x8xbf16>, tensor<8xbf16>, tensor<8xbf16>) -> (tensor<4x8xbf16>, tensor<8xbf16>, tensor<8xbf16>)
    return %output : tensor<4x8xbf16>
  }
}

// CHECK: stablehlo.all_gather
// CHECK: "stablehlo.batch_norm_training"
// CHECK-SAME: (tensor<4x4xbf16>, tensor<4xbf16>, tensor<4xbf16>) -> (tensor<4x4xbf16>, tensor<4xbf16>, tensor<4xbf16>)

// -----

// Test 3: batch_norm_grad, feature dim sharded (no data parallel).
// Scale is sharded on the feature dimension (kPassThrough), so sharding
// propagates without inserting any collective ops.
module @batch_norm_grad_feature_sharding attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  sdy.mesh @mesh = <["model"=2]>
  func.func @main(
    %arg0: tensor<4x8xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "input"},
    %arg1: tensor<8xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{"model"}]>, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "scale"},
    %arg2: tensor<8xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "mean"},
    %arg3: tensor<8xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "variance"},
    %arg4: tensor<4x8xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "grad_output"}
  ) -> tensor<4x8xbf16> {
    %grad_input, %grad_scale, %grad_bias = "stablehlo.batch_norm_grad"(%arg0, %arg1, %arg2, %arg3, %arg4) <{epsilon = 1.0e-5 : f32, feature_index = 1 : i64}> : (tensor<4x8xbf16>, tensor<8xbf16>, tensor<8xbf16>, tensor<8xbf16>, tensor<4x8xbf16>) -> (tensor<4x8xbf16>, tensor<8xbf16>, tensor<8xbf16>)
    return %grad_input : tensor<4x8xbf16>
  }
}

// CHECK-NOT: stablehlo.all_gather
// CHECK-NOT: stablehlo.all_reduce
// CHECK: "stablehlo.batch_norm_grad"
// CHECK-SAME: (tensor<4x4xbf16>, tensor<4xbf16>, tensor<4xbf16>, tensor<4xbf16>, tensor<4x4xbf16>)
// CHECK-SAME: -> (tensor<4x4xbf16>, tensor<4xbf16>, tensor<4xbf16>)

// -----

// Test 4: batch_norm_grad, batch dim + feature dim sharded (data parallel).
// Input and grad_output are batch-sharded (non-feature dim, kNeedReplication),
// so Shardy inserts all_gather to replicate across the batch axis before the op.
module @batch_norm_grad_data_parallel attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  sdy.mesh @mesh = <["model"=2, "batch"=4]>
  func.func @main(
    %arg0: tensor<4x8xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}]>, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "input"},
    %arg1: tensor<8xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{"model"}]>, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "scale"},
    %arg2: tensor<8xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "mean"},
    %arg3: tensor<8xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}]>, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "variance"},
    %arg4: tensor<4x8xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}]>, ttcore.argument_type = #ttcore.argument_type<input>, ttir.name = "grad_output"}
  ) -> tensor<4x8xbf16> {
    %grad_input, %grad_scale, %grad_bias = "stablehlo.batch_norm_grad"(%arg0, %arg1, %arg2, %arg3, %arg4) <{epsilon = 1.0e-5 : f32, feature_index = 1 : i64}> : (tensor<4x8xbf16>, tensor<8xbf16>, tensor<8xbf16>, tensor<8xbf16>, tensor<4x8xbf16>) -> (tensor<4x8xbf16>, tensor<8xbf16>, tensor<8xbf16>)
    return %grad_input : tensor<4x8xbf16>
  }
}

// CHECK: stablehlo.all_gather
// CHECK: "stablehlo.batch_norm_grad"
// CHECK-SAME: (tensor<4x4xbf16>, tensor<4xbf16>, tensor<4xbf16>, tensor<4xbf16>, tensor<4x4xbf16>)
// CHECK-SAME: -> (tensor<4x4xbf16>, tensor<4xbf16>, tensor<4xbf16>)
