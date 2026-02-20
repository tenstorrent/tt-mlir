// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-pipeline -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

sdy.mesh @mesh = <["model"=2]>

// Test 1: sharding on the feature dim (feature_index=1) must propagate to
// scale, bias, batch_mean, and batch_var (all 1-D, size=C).
func.func @batch_norm_training_feature_sharded(
    %input: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"model"}]>},
    %scale: tensor<8xf32>,
    %bias: tensor<8xf32>
) -> (tensor<4x8xf32>, tensor<8xf32>, tensor<8xf32>) {
  %output, %batch_mean, %batch_var = "stablehlo.batch_norm_training"(%input, %scale, %bias) {
      epsilon = 0.001 : f32, feature_index = 1 : i64
  } : (tensor<4x8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<4x8xf32>, tensor<8xf32>, tensor<8xf32>)
  return %output, %batch_mean, %batch_var : tensor<4x8xf32>, tensor<8xf32>, tensor<8xf32>
}

// CHECK-LABEL: func.func @batch_norm_training_feature_sharded
// CHECK: sdy.manual_computation(%arg0, %arg1, %arg2) in_shardings=[<@mesh, [{}, {"model"}]>, <@mesh, [{"model"}]>, <@mesh, [{"model"}]>] out_shardings=[<@mesh, [{}, {"model"}]>, <@mesh, [{"model"}]>, <@mesh, [{"model"}]>]
// CHECK: "stablehlo.batch_norm_training"({{.*}}) <{epsilon = {{.*}}, feature_index = 1 : i64}> : (tensor<4x4xf32>, tensor<4xf32>, tensor<4xf32>) -> (tensor<4x4xf32>, tensor<4xf32>, tensor<4xf32>)

// Test 2: sharding on a non-feature dim (dim 0, feature_index=1) propagates
// to output[0] but NOT to scale, bias, batch_mean, or batch_var.
func.func @batch_norm_training_non_feature_sharded(
    %input: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"model"}, {}]>},
    %scale: tensor<8xf32>,
    %bias: tensor<8xf32>
) -> (tensor<4x8xf32>, tensor<8xf32>, tensor<8xf32>) {
  %output, %batch_mean, %batch_var = "stablehlo.batch_norm_training"(%input, %scale, %bias) {
      epsilon = 0.001 : f32, feature_index = 1 : i64
  } : (tensor<4x8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<4x8xf32>, tensor<8xf32>, tensor<8xf32>)
  return %output, %batch_mean, %batch_var : tensor<4x8xf32>, tensor<8xf32>, tensor<8xf32>
}

// CHECK-LABEL: func.func @batch_norm_training_non_feature_sharded
// CHECK: sdy.manual_computation(%arg0, %arg1, %arg2) in_shardings=[<@mesh, [{"model"}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>] out_shardings=[<@mesh, [{"model"}, {}]>, <@mesh, [{}]>, <@mesh, [{}]>]
// CHECK: "stablehlo.batch_norm_training"({{.*}}) <{epsilon = {{.*}}, feature_index = 1 : i64}> : (tensor<2x8xf32>, tensor<8xf32>, tensor<8xf32>) -> (tensor<2x8xf32>, tensor<8xf32>, tensor<8xf32>)
