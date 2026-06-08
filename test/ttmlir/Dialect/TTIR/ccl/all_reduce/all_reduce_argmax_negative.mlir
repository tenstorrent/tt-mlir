// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --split-input-file --verify-diagnostics %s

// Mis-partitioned sharded-axis argmax: a per-shard ttir.argmax feeding a
// ttir.all_reduce<sum> over the integer index result. Summing per-shard
// indices does not produce a global argmax. See tt-mlir #8623.
module attributes {} {
  func.func @all_reduce_argmax_sum(%arg0: tensor<1x4096xbf16>) -> tensor<1xi32> {
    %0 = "ttir.argmax"(%arg0) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<1x4096xbf16>) -> tensor<1xi32>
    // expected-error @+1 {{'ttir.all_reduce' op Detected a mis-partitioned sharded-axis argmax}}
    %1 = "ttir.all_reduce"(%0) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<1xi32>) -> tensor<1xi32>
    return %1 : tensor<1xi32>
  }
}

// -----

module attributes {} {
  func.func @all_reduce_async_argmax_sum(%arg0: tensor<1x4096xbf16>) -> tensor<1xi32> {
    %0 = "ttir.argmax"(%arg0) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<1x4096xbf16>) -> tensor<1xi32>
    // expected-error @+1 {{'ttir.all_reduce_async' op Detected a mis-partitioned sharded-axis argmax}}
    %1 = "ttir.all_reduce_async"(%0) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<1xi32>) -> tensor<1xi32>
    return %1 : tensor<1xi32>
  }
}
