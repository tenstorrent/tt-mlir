// REQUIRES: stablehlo
// This file incorporates work covered by the following copyright and permission notice:
// SPDX-FileCopyrightText: Copyright (c) 2024 The Shardy Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --stablehlo-pipeline -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

sdy.mesh @mesh = <["model"=1, "batch"=2]>

func.func @concat_operands_are_results_of_slices(%arg0: tensor<4x40x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}, {}]>}, %arg1: tensor<4x60x256xf32>) -> tensor<4x80x256xf32> {
  %0 = stablehlo.slice %arg0 [0:4, 0:32, 0:256] : (tensor<4x40x256xf32>) -> tensor<4x32x256xf32>
  %1 = stablehlo.slice %arg1 [0:4, 0:48, 0:256] : (tensor<4x60x256xf32>) -> tensor<4x48x256xf32>
  %2 = stablehlo.concatenate %0, %1, dim = 1 : (tensor<4x32x256xf32>, tensor<4x48x256xf32>) -> tensor<4x80x256xf32>
  return %2 : tensor<4x80x256xf32>
}

// CHECK: sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh, [{"batch"}, {}, {}]>, <@mesh, [{"batch"}, {}, {}]>] out_shardings=[<@mesh, [{"batch"}, {}, {}]>]
// CHECK: stablehlo.slice %arg2 [0:2, 0:32, 0:256] : (tensor<2x40x256xf32>) -> tensor<2x32x256xf32>
// CHECK: stablehlo.slice %arg3 [0:2, 0:48, 0:256] : (tensor<2x60x256xf32>) -> tensor<2x48x256xf32>
// CHECK: sdy.return %3 : tensor<2x80x256xf32>

func.func @concat_operands_are_from_slices_of_the_same_tensor(%arg0: tensor<4x40x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}, {}]>}) -> tensor<4x96x256xf32> {
  %0 = stablehlo.slice %arg0 [0:4, 0:32, 0:256] : (tensor<4x40x256xf32>) -> tensor<4x32x256xf32>
  %1 = stablehlo.slice %arg0 [0:4, 0:24, 0:256] : (tensor<4x40x256xf32>) -> tensor<4x24x256xf32>
  %2 = stablehlo.concatenate %0, %arg0, %1, dim = 1 : (tensor<4x32x256xf32>, tensor<4x40x256xf32>, tensor<4x24x256xf32>) -> tensor<4x96x256xf32>
  return %2 : tensor<4x96x256xf32>
}

// CHECK: sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"batch"}, {}, {}]>] out_shardings=[<@mesh, [{"batch"}, {}, {}]>]
// CHECK: stablehlo.slice %arg1 [0:2, 0:32, 0:256] : (tensor<2x40x256xf32>) -> tensor<2x32x256xf32>
// CHECK: stablehlo.slice %arg1 [0:2, 0:24, 0:256] : (tensor<2x40x256xf32>) -> tensor<2x24x256xf32>
// CHECK: sdy.return %3 : tensor<2x96x256xf32>

func.func @non_batch_dim_slice(%arg0: tensor<4x40x256xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"batch"}, {}]>}, %arg1: tensor<4x60x256xf32>) -> tensor<4x80x256xf32> {
  %0 = stablehlo.slice %arg0 [0:4, 0:32, 0:256] : (tensor<4x40x256xf32>) -> tensor<4x32x256xf32>
  %1 = stablehlo.slice %arg1 [0:4, 0:48, 0:256] : (tensor<4x60x256xf32>) -> tensor<4x48x256xf32>
  %2 = stablehlo.concatenate %0, %1, dim = 1 : (tensor<4x32x256xf32>, tensor<4x48x256xf32>) -> tensor<4x80x256xf32>
  return %2 : tensor<4x80x256xf32>
}

// CHECK: sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh, [{}, {"batch"}, {}]>, <@mesh, [{}, {}, {}]>] out_shardings=[<@mesh, [{}, {}, {}]>]
// CHECK: stablehlo.all_gather
// CHECK: stablehlo.slice %1 [0:4, 0:32, 0:256]
// CHECK: stablehlo.slice %arg3 [0:4, 0:48, 0:256]
// CHECK: stablehlo.concatenate %2, %3
// CHECK: sdy.return %4 : tensor<4x80x256xf32>

func.func @partial_arg(%arg0: tensor<4x128x16384xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {"batch"}]>}) -> (tensor<4x128x8192xbf16>) {
  %1 = stablehlo.slice %arg0 [0:4, 0:128, 1:16384:2] : (tensor<4x128x16384xbf16>) -> tensor<4x128x8192xbf16>
  return %1 : tensor<4x128x8192xbf16>
}

// CHECK: sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{}, {}, {"batch"}]>] out_shardings=[<@mesh, [{}, {}, {}]>]
// CHECK: stablehlo.all_gather
// CHECK: stablehlo.slice %1 [0:4, 0:128, 1:16384:2]
// CHECK: sdy.return %2 : tensor<4x128x8192xbf16>

func.func @multiple_slice_unique_operand(%arg0: tensor<2560x9728xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}, {\22batch\22}]>"}, mhlo.sharding = "{devices=[1,2]<=[2]}", ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg1: tensor<19456x2560xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{\22batch\22}, {}]>"}, mhlo.sharding = "{devices=[2,1]<=[2]}", ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg2: tensor<2560xbf16> {mhlo.frontend_attributes = {xla.sdy.sharding = "#sdy.sharding<@mesh, [{}]>"}, mhlo.sharding = "{replicated}", ttcore.argument_type = #ttcore.argument_type<parameter>}) -> tensor<1x32x2560xbf16> {
  // CHECK-LABEL: func.func @multiple_slice_unique_operand(
  // CHECK: sdy.manual_computation(%arg0, %arg1, %arg2) in_shardings=[<@mesh, [{}, {"batch"}]>, <@mesh, [{"batch"}, {}]>, <@mesh, [{}]>] out_shardings=[<@mesh, [{}, {}, {}]>] manual_axes={"model", "batch"} (%arg3: tensor<2560x4864xbf16>, %arg4: tensor<9728x2560xbf16>, %arg5: tensor<2560xbf16>)
  // CHECK: %{{.*}} = stablehlo.slice %{{.*}} [0:1, 0:32, 0:4864] : (tensor<1x32x9728xbf16>) -> tensor<1x32x4864xbf16>
  // CHECK: %{{.*}} = stablehlo.slice %{{.*}} [0:1, 0:32, 4864:9728] : (tensor<1x32x9728xbf16>) -> tensor<1x32x4864xbf16>
  // CHECK: stablehlo.all_reduce
  // CHECK: sdy.return %{{.*}} : tensor<1x32x2560xbf16>
  %0 = stablehlo.reshape %arg2 : (tensor<2560xbf16>) -> tensor<1x1x2560xbf16>
  %1 = stablehlo.reshape %0 : (tensor<1x1x2560xbf16>) -> tensor<2560xbf16>
  %2 = stablehlo.broadcast_in_dim %1, dims = [2] : (tensor<2560xbf16>) -> tensor<1x32x2560xbf16>
  %3 = stablehlo.reshape %2 : (tensor<1x32x2560xbf16>) -> tensor<32x2560xbf16>
  %4 = stablehlo.reshape %arg1 : (tensor<19456x2560xbf16>) -> tensor<1x19456x2560xbf16>
  %5 = stablehlo.reshape %4 : (tensor<1x19456x2560xbf16>) -> tensor<19456x2560xbf16>
  %6 = stablehlo.transpose %5, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[2560,19456]{0,1}"} : (tensor<19456x2560xbf16>) -> tensor<2560x19456xbf16>
  %7 = stablehlo.dot_general %3, %6, contracting_dims = [1] x [0] : (tensor<32x2560xbf16>, tensor<2560x19456xbf16>) -> tensor<32x19456xbf16>
  %8 = stablehlo.reshape %7 : (tensor<32x19456xbf16>) -> tensor<1x32x19456xbf16>
  %9 = stablehlo.slice %8 [0:1, 0:32, 0:9728] : (tensor<1x32x19456xbf16>) -> tensor<1x32x9728xbf16>
  %10 = stablehlo.slice %8 [0:1, 0:32, 9728:19456] : (tensor<1x32x19456xbf16>) -> tensor<1x32x9728xbf16>
  %11 = stablehlo.multiply %9, %10 : tensor<1x32x9728xbf16>
  %12 = stablehlo.reshape %11 : (tensor<1x32x9728xbf16>) -> tensor<32x9728xbf16>
  %13 = stablehlo.transpose %arg0, dims = [1, 0] {result_layout = dense<[0, 1]> : tensor<2xindex>, xla_shape = "bf16[9728,2560]{0,1}"} : (tensor<2560x9728xbf16>) -> tensor<9728x2560xbf16>
  %14 = stablehlo.dot_general %12, %13, contracting_dims = [1] x [0] : (tensor<32x9728xbf16>, tensor<9728x2560xbf16>) -> tensor<32x2560xbf16>
  %15 = stablehlo.reshape %14 : (tensor<32x2560xbf16>) -> tensor<1x32x2560xbf16>
  return %15 : tensor<1x32x2560xbf16>
}
