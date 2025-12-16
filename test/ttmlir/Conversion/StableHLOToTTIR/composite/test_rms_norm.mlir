// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module @jit__rms_norm attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @empty_mesh = <["default_updated"=1, "default"=1]>

  // CHECK-LABEL: func.func public @test_rms_norm_no_weight
  // CHECK: "ttir.rms_norm"(%{{.*}})

  func.func public @test_rms_norm_no_weight(%arg0: tensor<4x32xf32> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>, ttir.name = "args_0"}) -> (tensor<4x32xf32> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
    %0 = stablehlo.reshape %arg0 : (tensor<4x32xf32>) -> tensor<1x4x32xf32>
    %1 = stablehlo.reshape %0 : (tensor<1x4x32xf32>) -> tensor<4x32xf32>
    %2 = stablehlo.composite "tenstorrent.rms_norm" %1 {composite_attributes = {normalized_shape = dense<32> : tensor<1xi64>}, decomposition = @tenstorrent.rms_norm.impl} : (tensor<4x32xf32>) -> tensor<4x32xf32>
    return %2 : tensor<4x32xf32>
  }

  func.func private @tenstorrent.rms_norm.impl(%arg0: tensor<4x32xf32> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>}) -> (tensor<4x32xf32> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
    %cst = stablehlo.constant dense<1.1920929E-7> : tensor<4x1xf32>
    %cst_0 = stablehlo.constant dense<3.125000e-02> : tensor<4xf32>
    %cst_1 = stablehlo.constant dense<2.000000e+00> : tensor<4x32xf32>
    %cst_2 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.power %arg0, %cst_1 : tensor<4x32xf32>
    %1 = stablehlo.reduce(%0 init: %cst_2) applies stablehlo.add across dimensions = [1] : (tensor<4x32xf32>, tensor<f32>) -> tensor<4xf32>
    %2 = stablehlo.multiply %1, %cst_0 : tensor<4xf32>
    %3 = stablehlo.reshape %2 : (tensor<4xf32>) -> tensor<4x1xf32>
    %4 = stablehlo.add %3, %cst : tensor<4x1xf32>
    %5 = stablehlo.rsqrt %4 : tensor<4x1xf32>
    %6 = stablehlo.reshape %5 : (tensor<4x1xf32>) -> tensor<4xf32>
    %7 = stablehlo.broadcast_in_dim %6, dims = [0] : (tensor<4xf32>) -> tensor<4x32xf32>
    %8 = stablehlo.multiply %arg0, %7 : tensor<4x32xf32>
    return %8 : tensor<4x32xf32>
  }
}
