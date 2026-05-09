// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

// Tenstorrent gated-activation composites (swiglu/glu/geglu/reglu) legalize
// to ttir.gated_activation, carrying the activation string and dim attr.

module @jit__gated_activation attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  sdy.mesh @empty_mesh = <["default_updated"=1, "default"=1]>

  // CHECK-LABEL: func.func public @test_swiglu
  // CHECK: "ttir.gated_activation"
  // CHECK-SAME: activation = "swiglu"
  // CHECK-SAME: dim = -1
  func.func public @test_swiglu(%arg0: tensor<4x64xf32> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>}) -> (tensor<4x32xf32> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
    %0 = stablehlo.composite "tenstorrent.swiglu" %arg0 {composite_attributes = {dim = -1 : i32}, decomposition = @tenstorrent.swiglu.impl} : (tensor<4x64xf32>) -> tensor<4x32xf32>
    return %0 : tensor<4x32xf32>
  }

  // CHECK-LABEL: func.func public @test_glu
  // CHECK: "ttir.gated_activation"
  // CHECK-SAME: activation = "glu"
  // CHECK-SAME: dim = -1
  func.func public @test_glu(%arg0: tensor<4x64xf32> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>}) -> (tensor<4x32xf32> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
    %0 = stablehlo.composite "tenstorrent.glu" %arg0 {composite_attributes = {dim = -1 : i32}, decomposition = @tenstorrent.glu.impl} : (tensor<4x64xf32>) -> tensor<4x32xf32>
    return %0 : tensor<4x32xf32>
  }

  // CHECK-LABEL: func.func public @test_geglu
  // CHECK: "ttir.gated_activation"
  // CHECK-SAME: activation = "geglu"
  func.func public @test_geglu(%arg0: tensor<4x64xf32> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>}) -> (tensor<4x32xf32> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
    %0 = stablehlo.composite "tenstorrent.geglu" %arg0 {composite_attributes = {dim = -1 : i32}, decomposition = @tenstorrent.geglu.impl} : (tensor<4x64xf32>) -> tensor<4x32xf32>
    return %0 : tensor<4x32xf32>
  }

  // CHECK-LABEL: func.func public @test_reglu
  // CHECK: "ttir.gated_activation"
  // CHECK-SAME: activation = "reglu"
  func.func public @test_reglu(%arg0: tensor<4x64xf32> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>}) -> (tensor<4x32xf32> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
    %0 = stablehlo.composite "tenstorrent.reglu" %arg0 {composite_attributes = {dim = -1 : i32}, decomposition = @tenstorrent.reglu.impl} : (tensor<4x64xf32>) -> tensor<4x32xf32>
    return %0 : tensor<4x32xf32>
  }

  // Decompositions are stand-ins for the real lowerings; the legalize pass
  // never inspects them, but stablehlo.composite requires the symbol to exist.
  func.func private @tenstorrent.swiglu.impl(%arg0: tensor<4x64xf32> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>}) -> (tensor<4x32xf32> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<4x32xf32>
    return %cst : tensor<4x32xf32>
  }
  func.func private @tenstorrent.glu.impl(%arg0: tensor<4x64xf32> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>}) -> (tensor<4x32xf32> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<4x32xf32>
    return %cst : tensor<4x32xf32>
  }
  func.func private @tenstorrent.geglu.impl(%arg0: tensor<4x64xf32> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>}) -> (tensor<4x32xf32> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<4x32xf32>
    return %cst : tensor<4x32xf32>
  }
  func.func private @tenstorrent.reglu.impl(%arg0: tensor<4x64xf32> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<unsharded>}) -> (tensor<4x32xf32> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<4x32xf32>
    return %cst : tensor<4x32xf32>
  }
}
