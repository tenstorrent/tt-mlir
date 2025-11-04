// REQUIRES: stablehlo
// This file incorporates work covered by the following copyright and permission notice:
// SPDX-FileCopyrightText: Copyright (c) 2024 The Shardy Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --stablehlo-pipeline -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

module @SyncTensorsGraph.27 attributes {mhlo.cross_program_prefetches = [], mhlo.frontend_attributes = {xla.sdy.meshes = "{mesh = #sdy.mesh<[\22_axis_0\22=2]>}"}, mhlo.input_output_alias = [], mhlo.is_dynamic = false, mhlo.use_auto_spmd_partitioning = false} {
  func.func @main(%arg0: tensor<2048xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}]>"}, mhlo.sharding = "{replicated}", ttcore.argument_type = #ttcore.argument_type<input>} loc("p0.1"), %arg1: tensor<2048x2048xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22_axis_0\22}]>"}, mhlo.sharding = "{devices=[1,2]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>} loc("p1.2"), %arg2: tensor<2048xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{\22_axis_0\22}]>"}, mhlo.sharding = "{devices=[2]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>} loc("p2.4"), %arg3: tensor<2048x2048xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{\22_axis_0\22}, {}]>"}, mhlo.sharding = "{devices=[2,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<input>} loc("p3.5"), %arg4: tensor<2048x2048xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {}]>"}, mhlo.sharding = "{replicated}", ttcore.argument_type = #ttcore.argument_type<input>}) -> tensor<2048x2048xbf16> {
    %cst = stablehlo.constant dense<0.000000e+00> : tensor<bf16>
    // CHECK: %[[CST:.*]] = stablehlo.constant
    %0 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<bf16>) -> tensor<2048x2048xbf16>
    // CHECK: %[[B_FULL:.*]] = stablehlo.broadcast_in_dim %[[CST]]{{.*}}
    %1 = stablehlo.transpose %arg3, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,2048]{0,1}"} : (tensor<2048x2048xbf16>) -> tensor<2048x2048xbf16>
    %2 = stablehlo.dot_general %arg4, %1, contracting_dims = [1] x [0] : (tensor<2048x2048xbf16>, tensor<2048x2048xbf16>) -> tensor<2048x2048xbf16>
    %3 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<2048xbf16>) -> tensor<2048x2048xbf16>
    %4 = stablehlo.add %2, %3 : tensor<2048x2048xbf16>
    // CHECK: %[[MAX1_IN:.*]] = stablehlo.add
    // CHECK: %[[B_MID:.*]] = stablehlo.broadcast_in_dim %[[CST]]{{.*}}
    // CHECK: %[[MAX1:.*]] = stablehlo.maximum %[[MAX1_IN]], %[[B_MID]]
    %5 = stablehlo.maximum %4, %0 : tensor<2048x2048xbf16>
    %6 = stablehlo.transpose %arg1, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2048,2048]{0,1}"} : (tensor<2048x2048xbf16>) -> tensor<2048x2048xbf16>
    %7 = stablehlo.dot_general %5, %6, contracting_dims = [1] x [0] : (tensor<2048x2048xbf16>, tensor<2048x2048xbf16>) -> tensor<2048x2048xbf16>
    %8 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<2048xbf16>) -> tensor<2048x2048xbf16>
    %9 = stablehlo.add %7, %8 : tensor<2048x2048xbf16>
    // CHECK: %[[MAX2_IN:.*]] = stablehlo.add %{{.*}}, %{{.*}} : tensor<2048x2048xbf16>
    // CHECK: %{{.*}} = stablehlo.maximum %[[MAX2_IN]], %[[B_FULL]]
    %10 = stablehlo.maximum %9, %0 : tensor<2048x2048xbf16>
    return %10 : tensor<2048x2048xbf16>
  }
}
