// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

// Composite-with-residual: tenstorrent.rms_norm carrying `has_residual = true`
// must legalize to ttir.rms_norm with the new optional residual operand set.
// The 4-element operandSegmentSizes records [input, weight, bias, residual]
// with weight=1, bias=0, residual=1.

module @jit__residual_rms_norm attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @empty_mesh = <["default_updated"=1, "default"=1]>

  // CHECK-LABEL: func.func public @test_rms_norm_with_residual
  // CHECK: "ttir.rms_norm"
  // CHECK-SAME: operandSegmentSizes = array<i32: 1, 1, 0, 1>

  func.func public @test_rms_norm_with_residual(
      %arg0: tensor<32xf32> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>},
      %arg1: tensor<4x32xf32> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "args_0"},
      %arg2: tensor<4x32xf32> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "args_1"})
      -> (tensor<4x32xf32> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
    %0 = stablehlo.composite "tenstorrent.rms_norm" %arg1, %arg0, %arg2 {composite_attributes = {has_residual = true, normalized_shape = dense<32> : tensor<1xi64>}, decomposition = @tenstorrent.rms_norm_residual.impl} : (tensor<4x32xf32>, tensor<32xf32>, tensor<4x32xf32>) -> tensor<4x32xf32>
    return %0 : tensor<4x32xf32>
  }

  func.func private @tenstorrent.rms_norm_residual.impl(
      %arg0: tensor<4x32xf32> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>},
      %arg1: tensor<32xf32> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "args_w"},
      %arg2: tensor<4x32xf32> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "args_r"})
      -> (tensor<4x32xf32> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
    %0 = stablehlo.add %arg0, %arg2 : tensor<4x32xf32>
    %cst = stablehlo.constant dense<1.1920929E-7> : tensor<4x1xf32>
    %cst_0 = stablehlo.constant dense<3.125000e-02> : tensor<4xf32>
    %cst_1 = stablehlo.constant dense<2.000000e+00> : tensor<4x32xf32>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %1 = stablehlo.power %0, %cst_1 : tensor<4x32xf32>
    %2 = stablehlo.reduce(%1 init: %cst_2) applies stablehlo.add across dimensions = [1] : (tensor<4x32xf32>, tensor<f32>) -> tensor<4xf32>
    %3 = stablehlo.multiply %2, %cst_0 : tensor<4xf32>
    %4 = stablehlo.reshape %3 : (tensor<4xf32>) -> tensor<4x1xf32>
    %5 = stablehlo.add %4, %cst : tensor<4x1xf32>
    %6 = stablehlo.rsqrt %5 : tensor<4x1xf32>
    %7 = stablehlo.reshape %6 : (tensor<4x1xf32>) -> tensor<4xf32>
    %8 = stablehlo.broadcast_in_dim %7, dims = [0] : (tensor<4xf32>) -> tensor<4x32xf32>
    %9 = stablehlo.multiply %0, %8 : tensor<4x32xf32>
    %10 = stablehlo.broadcast_in_dim %arg1, dims = [1] : (tensor<32xf32>) -> tensor<4x32xf32>
    %11 = stablehlo.multiply %9, %10 : tensor<4x32xf32>
    return %11 : tensor<4x32xf32>
  }
}
