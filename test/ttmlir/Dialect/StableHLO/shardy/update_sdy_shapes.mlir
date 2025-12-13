// REQUIRES: stablehlo
// This file incorporates work covered by the following copyright and permission notice:
// SPDX-FileCopyrightText: Copyright (c) 2024 The Shardy Authors
// SPDX-License-Identifier: Apache-2.0

// RUN: rm -rf %t.mlir
// RUN: ttmlir-opt --update-global-to-local-shapes -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

sdy.mesh @mesh = <["x"=2, "y"=4]>

func.func @sdy_shardings_only(%arg0: tensor<1024x8192xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<1x1024x8192xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}, {"y"}]>}) {
    %0:1 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}, {"y"}]>] out_shardings=[<@mesh, [{}, {"x"}, {"y"}]>] manual_axes = {} (%arg1: tensor<1024x8192xbf16>) {
        %1 = stablehlo.reshape %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"x"}, {"y"}]>]>} : (tensor<1024x8192xbf16>) -> tensor<1x1024x8192xbf16>
        sdy.return %1 : tensor<1x1024x8192xbf16>
    } : (tensor<1024x8192xbf16>) -> (tensor<1x1024x8192xbf16>)
    return %0#0 : tensor<1x1024x8192xbf16>
}
// CHECK: %1 = stablehlo.reshape %arg1 : (tensor<512x2048xbf16>) -> tensor<1x512x2048xbf16>
// CHECK: sdy.return %1 : tensor<1x512x2048xbf16>

func.func @sdy_ccls_only(%arg0: tensor<8192xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}]>}, %arg1: tensor<7x1024xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) -> (tensor<8192xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{"y"}]>}, tensor<7x1024xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"x"}]>}) {
    %0:2 = sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh, [{}]>, <@mesh, [{}, {"x"}]>] out_shardings=[<@mesh, [{"y"}]>, <@mesh, [{}, {"x"}]>] manual_axes = {} (%arg2: tensor<8192xbf16>, %arg3: tensor<7x1024xbf16>) {
        %1 = sdy.all_slice [{"y"}] %arg2 out_sharding=<@mesh, [{"y"}]> : tensor<8192xbf16>
        %2 = sdy.all_reduce {"y"} %arg3 out_sharding=<@mesh, [{}, {"x"}]> : tensor<7x1024xbf16>
        sdy.return %1, %2 : tensor<8192xbf16>, tensor<7x1024xbf16>
    } : (tensor<8192xbf16>, tensor<7x1024xbf16>) -> (tensor<8192xbf16>, tensor<7x1024xbf16>)
    return %0#0, %0#1 : tensor<8192xbf16>, tensor<7x1024xbf16>
}
// %4 is the result of all_slice on %arg0
// %5 is the result of all_reduce on %arg1
// CHECK: sdy.return %4, %5 : tensor<2048xbf16>, tensor<7x512xbf16>

func.func @sharded_matmul(%arg0: tensor<544x8192xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, %arg1: tensor<8192x1024xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) -> (tensor<544x1024xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {"y"}]>}) {
    %0:4 = sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh, [{"x"}, {}]>, <@mesh, [{"x"}, {"y"}]>] out_shardings=[<@mesh, [{}, {"x"}]>, <@mesh, [{}, {"y"}]>, <@mesh, [{}, {"y"}]>, <@mesh, [{"x"}, {"y"}]>] manual_axes = {} (%arg2: tensor<544x8192xbf16>, %arg3: tensor<8192x1024xbf16>) {
        %1 = sdy.all_to_all [{"x"}: 0->1] %arg2 out_sharding=<@mesh, [{}, {"x"}]> : tensor<544x8192xbf16>
        %2 = stablehlo.dot_general %1, %arg3, contracting_dims = [1] x [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{}, {"y"}]>]>} : (tensor<544x8192xbf16>, tensor<8192x1024xbf16>) -> tensor<544x1024xbf16>
        %3 = sdy.all_reduce {"x"} %2 out_sharding=<@mesh, [{}, {"y"}]> : tensor<544x1024xbf16>
        %4 = sdy.all_slice [{"x"}, {}] %3 out_sharding=<@mesh, [{"x"}, {"y"}]> : tensor<544x1024xbf16>
        sdy.return %1, %2, %3, %4 : tensor<544x8192xbf16>, tensor<544x1024xbf16>, tensor<544x1024xbf16>, tensor<544x1024xbf16>
    } : (tensor<544x8192xbf16>, tensor<8192x1024xbf16>) -> (tensor<544x8192xbf16>, tensor<544x1024xbf16>, tensor<544x1024xbf16>, tensor<544x1024xbf16>)
    return %0#3 : tensor<544x1024xbf16>
}

// %1 is the result of the sdy.all_to_all
// %2 is the result of the stablehlo.dot_general
// %3 is the result of the sdy.all_reduce
// %4 is the result of the sdy.all_slice
// CHECK: sdy.return %1, %2, %3, %7 : tensor<544x4096xbf16>, tensor<544x256xbf16>, tensor<544x256xbf16>, tensor<272x256xbf16>
