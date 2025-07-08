// REQUIRES: stablehlo
// This file incorporates work covered by the following copyright and permission notice:
// SPDX-FileCopyrightText: Copyright (c) 2024 The Shardy Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-pipeline="mesh-shape=2,4" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir

sdy.mesh @mesh = <["model"=2, "batch"=4]>

func.func public @sharding_mismatch(%arg0: tensor<8192x784xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}]>}, %arg1: tensor<784x2048xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}]>}, %arg2: tensor<8192x2048xf32>) -> (tensor<8192x2048xf32> {jax.result_info = ""}) {
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<8192x784xf32>, tensor<784x2048xf32>) -> tensor<8192x2048xf32>
  %1 = stablehlo.add %0, %arg2 : tensor<8192x2048xf32>
  return %1 : tensor<8192x2048xf32>
}

// CHECK: sdy.manual_computation(%arg0, %arg1, %arg2) in_shardings=[<@mesh, [{"batch"}, {}]>, <@mesh, [{"batch"}, {}]>, <@mesh, [{"batch"}, {}]>] out_shardings=[<@mesh, [{"batch"}, {}]>]
// CHECK: stablehlo.all_gather

func.func public @sharding_constraint(%arg0: tensor<32x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"model"}, {"batch"}]>}) -> (tensor<32x32xf32> {jax.result_info = "", sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
  %0 = sdy.sharding_constraint %arg0 <@mesh, [{}, {}]> : tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}

// CHECK: sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"model"}, {"batch"}]>] out_shardings=[<@mesh, [{}, {}]>]
// CHECK: stablehlo.all_gather
// CHECK: stablehlo.all_gather
// CHECK: sdy.return %2 : tensor<32x32xf32>

func.func @reduce_scatter(%arg0 : tensor<16x16xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"model"}, {}]>}) -> tensor<16x16xf32> {
  %0 = sdy.reduce_scatter [{}, {"batch"}] %arg0 out_sharding=<@mesh, [{"model"}, {"batch"}]> : tensor<16x16xf32>
  return %0 : tensor<16x16xf32>
}

// CHECK: sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"model"}, {}]>] out_shardings=[<@mesh, [{"model"}, {"batch"}]>]
// CHECK: stablehlo.reduce_scatter
// CHECK: sdy.return %1 : tensor<8x4xf32>

func.func public @several_collectives(%arg0: tensor<1024x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {"model"}]>}, %arg1: tensor<256x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {"model"}]>}, %arg2: tensor<256x256xf32>) -> (tensor<1024x256xf32> {jax.result_info = "result"}) {
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1024x256xf32>, tensor<256x256xf32>) -> tensor<1024x256xf32>
  %1 = stablehlo.dot_general %0, %arg2, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1024x256xf32>, tensor<256x256xf32>) -> tensor<1024x256xf32>
  return %1 : tensor<1024x256xf32>
}

// CHECK: sdy.manual_computation(%arg0, %arg1, %arg2) in_shardings=[<@mesh, [{"batch"}, {"model"}]>, <@mesh, [{"batch"}, {"model"}]>, <@mesh, [{"model"}, {}]>] out_shardings=[<@mesh, [{"batch"}, {}]>]
// CHECK: stablehlo.all_gather
// CHECK: stablehlo.all_gather
// CHECK: stablehlo.all_reduce
// CHECK: stablehlo.add
// CHECK: sdy.return %5 : tensor<256x256xf32>

func.func @dot_general_no_batching_dims(%arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {"model"}]>}, %arg1: tensor<32x16xf32>) -> tensor<8x16xf32> {
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK: sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh, [{"batch"}, {"model"}]>, <@mesh, [{"model"}, {}]>] out_shardings=[<@mesh, [{"batch"}, {}]>]
// CHECK: stablehlo.all_reduce
// CHECK: sdy.return %2 : tensor<2x16xf32>

func.func @reduce_single_result_reduction_dim_sharded(%arg0: tensor<2x64x13xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"batch"}, {}]>}) -> tensor<2x13xf32> {
  %0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
  %1 = stablehlo.reduce(%arg0 init: %0) applies stablehlo.add across dimensions = [1] : (tensor<2x64x13xf32>, tensor<f32>) -> tensor<2x13xf32>
  return %1 : tensor<2x13xf32>
}

// CHECK: sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}, {"batch"}, {}]>] out_shardings=[<@mesh, [{}, {}]>]
// CHECK: stablehlo.all_reduce
// CHECK: sdy.return %2 : tensor<2x13xf32>

func.func @dot_compatible_contracting_dim_sharded(
    %arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {"model"}]>},
    %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"model"}, {}]>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}]>}) {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"batch"}, {}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// CHECK: sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh, [{"batch"}, {"model"}]>, <@mesh, [{"model"}, {}]>] out_shardings=[<@mesh, [{"batch"}, {}]>]
// CHECK: stablehlo.all_reduce
// CHECK: sdy.return %2 : tensor<2x16xf32>

func.func @all_to_all_single_axis(%arg0 : tensor<8x8x8xf32> {sdy.sharding=#sdy.sharding<@mesh, [{"batch"}, {"model"}, {}]>}) -> tensor<8x8x8xf32> {
  %0 = sdy.reshard %arg0 <@mesh, [{}, {"model"}, {"batch"}]> : tensor<8x8x8xf32>
  return %0 : tensor<8x8x8xf32>
}

// CHECK: sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"batch"}, {"model"}, {}]>] out_shardings=[<@mesh, [{}, {"model"}, {"batch"}]>]
// CHECK: stablehlo.all_to_all
// CHECK: sdy.return %1 : tensor<8x4x2xf32>
