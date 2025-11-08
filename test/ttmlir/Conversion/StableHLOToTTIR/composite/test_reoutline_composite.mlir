// REQUIRES: stablehlo
// RUN: ttmlir-opt -split-input-file --reoutline-composite --sdy-close-shardings --canonicalize --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module @CompositeBackSuccess attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  sdy.mesh @mesh = <["_axis_0_updated"=1, "_axis_0"=2]>
  func.func @main(%arg0: tensor<32x32xf32> {ttcore.shard_status = #ttcore.shard_status<presharded>}) -> (tensor<32x32xf32> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}, {"_axis_0"}]>] out_shardings=[<@mesh, [{?}, {"_axis_0", ?}]>] manual_axes={"_axis_0_updated", "_axis_0"} (%arg1: tensor<32x16xf32>) {
        %1 = stablehlo.reshape %arg1 : (tensor<32x16xf32>) -> tensor<1x32x16xf32>
        %2 = stablehlo.reshape %1 : (tensor<1x32x16xf32>) -> tensor<32x16xf32>
        // CHECK: ttir.gelu
        // CHECK-SAME: (tensor<32x16xf32>, tensor<32x16xf32>) -> tensor<32x16xf32>
        %cst_4 = stablehlo.constant {reoutline.comp_attrs = {approximate = "tanh"}, reoutline.group = "composite_tenstorrent.gelu_tanh.impl", reoutline.orig_name = "tenstorrent.gelu_tanh", reoutline.seed} dense<5.000000e-01> : tensor<f32>
        %cst_5 = stablehlo.constant {reoutline.group = "composite_tenstorrent.gelu_tanh.impl"} dense<1.000000e+00> : tensor<f32>
        %cst_6 = stablehlo.constant {reoutline.group = "composite_tenstorrent.gelu_tanh.impl"} dense<0.797884583> : tensor<f32>
        %cst_7 = stablehlo.constant {reoutline.group = "composite_tenstorrent.gelu_tanh.impl"} dense<4.471500e-02> : tensor<f32>
        %cst_8 = stablehlo.constant {reoutline.group = "composite_tenstorrent.gelu_tanh.impl"} dense<3.000000e+00> : tensor<f32>
        %16 = stablehlo.broadcast_in_dim %cst_8, dims = [] {reoutline.group = "composite_tenstorrent.gelu_tanh.impl"} : (tensor<f32>) -> tensor<32x16xf32>
        %17 = stablehlo.broadcast_in_dim %cst_7, dims = [] {reoutline.group = "composite_tenstorrent.gelu_tanh.impl"} : (tensor<f32>) -> tensor<32x16xf32>
        %18 = stablehlo.broadcast_in_dim %cst_6, dims = [] {reoutline.group = "composite_tenstorrent.gelu_tanh.impl"} : (tensor<f32>) -> tensor<32x16xf32>
        %19 = stablehlo.broadcast_in_dim %cst_5, dims = [] {reoutline.group = "composite_tenstorrent.gelu_tanh.impl"} : (tensor<f32>) -> tensor<32x16xf32>
        %20 = stablehlo.broadcast_in_dim %cst_4, dims = [] {reoutline.group = "composite_tenstorrent.gelu_tanh.impl"} : (tensor<f32>) -> tensor<32x16xf32>
        %21 = stablehlo.multiply %20, %2 {reoutline.group = "composite_tenstorrent.gelu_tanh.impl"} : tensor<32x16xf32>
        %22 = stablehlo.power %2, %16 {reoutline.group = "composite_tenstorrent.gelu_tanh.impl"} : tensor<32x16xf32>
        %23 = stablehlo.multiply %22, %17 {reoutline.group = "composite_tenstorrent.gelu_tanh.impl"} : tensor<32x16xf32>
        %24 = stablehlo.add %2, %23 {reoutline.group = "composite_tenstorrent.gelu_tanh.impl"} : tensor<32x16xf32>
        %25 = stablehlo.multiply %24, %18 {reoutline.group = "composite_tenstorrent.gelu_tanh.impl"} : tensor<32x16xf32>
        %26 = stablehlo.tanh %25 {reoutline.group = "composite_tenstorrent.gelu_tanh.impl"} : tensor<32x16xf32>
        %27 = stablehlo.add %26, %19 {reoutline.group = "composite_tenstorrent.gelu_tanh.impl"} : tensor<32x16xf32>
        %28 = stablehlo.multiply %21, %27 {reoutline.group = "composite_tenstorrent.gelu_tanh.impl"} : tensor<32x16xf32>
    sdy.return %28 : tensor<32x16xf32>
    } : (tensor<32x32xf32>) -> tensor<32x32xf32>
    return %0 : tensor<32x32xf32>
  }
}

// -----

module @CompositeBackFailed attributes {mhlo.cross_program_prefetches = [], mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  sdy.mesh @mesh = <["_axis_0_updated"=1, "_axis_0"=2]>
  func.func @main(%arg0: tensor<32x32xf32> {ttcore.shard_status = #ttcore.shard_status<presharded>}) -> (tensor<32x32xf32> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}, {"_axis_0"}]>] out_shardings=[<@mesh, [{?}, {"_axis_0", ?}]>] manual_axes={"_axis_0_updated", "_axis_0"} (%arg1: tensor<32x16xf32>) {
        %1 = stablehlo.reshape %arg1 : (tensor<32x16xf32>) -> tensor<1x32x16xf32>
        %2 = stablehlo.reshape %1 : (tensor<1x32x16xf32>) -> tensor<32x16xf32>
        // CHECK: ttir.tanh
        // CHECK-NOT: ttir.gelu
        // CHECK-SAME: (tensor<32x16xf32>, tensor<32x16xf32>) -> tensor<32x16xf32>
        %cst_4 = stablehlo.constant {reoutline.comp_attrs = {approximate = "tanh"}, reoutline.group = "composite_tenstorrent.gelu_tanh.impl", reoutline.orig_name = "tenstorrent.gelu_tanh", reoutline.seed} dense<5.000000e-01> : tensor<f32>
        %cst_5 = stablehlo.constant {reoutline.group = "composite_tenstorrent.gelu_tanh.impl"} dense<1.000000e+00> : tensor<f32>
        %cst_6 = stablehlo.constant {reoutline.group = "composite_tenstorrent.gelu_tanh.impl"} dense<0.797884583> : tensor<f32>
        %cst_7 = stablehlo.constant {reoutline.group = "composite_tenstorrent.gelu_tanh.impl"} dense<4.471500e-02> : tensor<f32>
        %cst_8 = stablehlo.constant {reoutline.group = "composite_tenstorrent.gelu_tanh.impl"} dense<3.000000e+00> : tensor<f32>
        %16 = stablehlo.broadcast_in_dim %cst_8, dims = [] {reoutline.group = "composite_tenstorrent.gelu_tanh.impl"} : (tensor<f32>) -> tensor<32x16xf32>
        %17 = stablehlo.broadcast_in_dim %cst_7, dims = [] {reoutline.group = "composite_tenstorrent.gelu_tanh.impl"} : (tensor<f32>) -> tensor<32x16xf32>
        %18 = stablehlo.broadcast_in_dim %cst_6, dims = [] {reoutline.group = "composite_tenstorrent.gelu_tanh.impl"} : (tensor<f32>) -> tensor<32x16xf32>
        %19 = stablehlo.broadcast_in_dim %cst_5, dims = [] {reoutline.group = "composite_tenstorrent.gelu_tanh.impl"} : (tensor<f32>) -> tensor<32x16xf32>
        %20 = stablehlo.broadcast_in_dim %cst_4, dims = [] {reoutline.group = "composite_tenstorrent.gelu_tanh.impl"} : (tensor<f32>) -> tensor<32x16xf32>
        %21 = stablehlo.multiply %20, %2 {reoutline.group = "composite_tenstorrent.gelu_tanh.impl"} : tensor<32x16xf32>
        %22 = stablehlo.power %2, %16 {reoutline.group = "composite_tenstorrent.gelu_tanh.impl"} : tensor<32x16xf32>
        %23 = stablehlo.multiply %22, %17 {reoutline.group = "composite_tenstorrent.gelu_tanh.impl"} : tensor<32x16xf32>
        %24 = stablehlo.add %2, %23 {reoutline.group = "composite_tenstorrent.gelu_tanh.impl"} : tensor<32x16xf32>
        %25 = stablehlo.multiply %24, %18 {reoutline.group = "composite_tenstorrent.gelu_tanh.impl"} : tensor<32x16xf32>
        %26 = stablehlo.tanh %25 {reoutline.group = "composite_tenstorrent.gelu_tanh.impl"} : tensor<32x16xf32>
        %27 = stablehlo.add %26, %19 {reoutline.group = "composite_tenstorrent.gelu_tanh.impl"} : tensor<32x16xf32>
        // If some instructions were introduced between flattening and reoutlining, the reoutline will fail.
        // All other instructions are same above success case, except the following 2 lines:
        %28 = stablehlo.reshape %27 : (tensor<32x16xf32>) -> tensor<1x32x16xf32>
        %29 = stablehlo.reshape %28 : (tensor<1x32x16xf32>) -> tensor<32x16xf32>
        %30 = stablehlo.multiply %21, %29 {reoutline.group = "composite_tenstorrent.gelu_tanh.impl"} : tensor<32x16xf32>
    sdy.return %30 : tensor<32x16xf32>
    } : (tensor<32x32xf32>) -> tensor<32x32xf32>
    return %0 : tensor<32x32xf32>
  }
}
