// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-pipeline -o %t.mlir %s
// RUN: ttmlir-opt --legalize-stablehlo-composite-to-ttir -o %t2.mlir %t.mlir
// RUN: FileCheck %s --input-file=%t2.mlir

// Tests the distributed topk lowering for vocab-sharded TP inference.
//
// With vocab sharded across 4 chips (vocab=65536 per chip), the composite
// must lower to:
//   1. local ttir.topk on the shard  → [1, k] values + local indices
//   2. ttir.all_gather × 2           → [1, k*N] values + indices
//   3. ttir.constant + ttir.add      → [1, k*N] global indices
//   4. ttir.topk (merge)             → [1, k] merged values + sort order
//   5. ttir.gather                   → [1, k] aligned global indices
//
// The non-sharded path (single chip, vocab replicated) must lower to a
// plain ttir.topk without any all_gather.

sdy.mesh @mesh = <["batch"=1, "vocab"=4]>

// -----------------------------------------------------------------------
// Sharded case: vocab dim split 4 ways, k=32, input [1, 65536] global
// (each chip sees [1, 16384] local after sharding pipeline).
// -----------------------------------------------------------------------
// CHECK-LABEL: func.func public @topk_vocab_sharded
// CHECK: sdy.manual_computation
// Step 1: local topk on the shard [1, 16384] → [1, 32]
// CHECK: "ttir.topk"{{.*}}tensor<1x16384xf32>
// Step 2: all-gather values and indices → [1, 128]
// CHECK: "ttir.all_gather"{{.*}}tensor<1x128xf32>
// CHECK: "ttir.all_gather"{{.*}}tensor<1x128xi32>
// Step 3: compile-time shard offset + add → global indices [1, 128]
// CHECK: "ttir.constant"{{.*}}tensor<1x128xi32>
// CHECK: "ttir.add"{{.*}}tensor<1x128xi32>
// Step 4: merge topk on [1, 128] candidates → [1, 32]
// CHECK: "ttir.topk"{{.*}}tensor<1x128xf32>
// Step 5: gather to align global indices with sort order → [1, 32]
// CHECK: "ttir.gather"{{.*}}tensor<1x32xi32>
func.func public @topk_vocab_sharded(
    %arg0: tensor<1x65536xf32>
        {sdy.sharding = #sdy.sharding<@mesh, [{}, {"vocab"}]>})
    -> (tensor<1x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
        tensor<1x32xi64> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}) {
  %0:2 = stablehlo.composite "tenstorrent.topk" %arg0 {
      composite_attributes = {dim = -1 : i64, k = 32 : i64,
                              largest = true, sorted = true},
      decomposition = @tenstorrent.topk.impl
  } : (tensor<1x65536xf32>) -> (tensor<1x32xf32>, tensor<1x32xi64>)
  return %0#0, %0#1 : tensor<1x32xf32>, tensor<1x32xi64>
}

// -----------------------------------------------------------------------
// Non-sharded case: vocab replicated (batch sharded instead).
// Must produce a plain ttir.topk with no all_gather.
// -----------------------------------------------------------------------
// CHECK-LABEL: func.func public @topk_vocab_replicated
// CHECK: sdy.manual_computation
// CHECK: "ttir.topk"
// CHECK-NOT: ttir.all_gather
// CHECK: sdy.return
func.func public @topk_vocab_replicated(
    %arg0: tensor<1x65536xf32>
        {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}]>})
    -> (tensor<1x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}]>},
        tensor<1x32xi64> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}]>}) {
  %0:2 = stablehlo.composite "tenstorrent.topk" %arg0 {
      composite_attributes = {dim = -1 : i64, k = 32 : i64,
                              largest = true, sorted = true},
      decomposition = @tenstorrent.topk.impl
  } : (tensor<1x65536xf32>) -> (tensor<1x32xf32>, tensor<1x32xi64>)
  return %0#0, %0#1 : tensor<1x32xf32>, tensor<1x32xi64>
}

func.func private @tenstorrent.topk.impl(%arg0: tensor<1x65536xf32>)
    -> (tensor<1x32xf32>, tensor<1x32xi64>) {
  %0 = stablehlo.iota dim = 0 : tensor<65536xi32>
  %1 = stablehlo.reshape %0 : (tensor<65536xi32>) -> tensor<1x65536xi32>
  %2:2 = "stablehlo.sort"(%arg0, %1) <{dimension = 1 : i64}> ({
  ^bb0(%a: tensor<f32>, %b: tensor<f32>, %c: tensor<i32>, %d: tensor<i32>):
    %cmp = stablehlo.compare GT, %a, %b, TOTALORDER : (tensor<f32>, tensor<f32>) -> tensor<i1>
    stablehlo.return %cmp : tensor<i1>
  }) : (tensor<1x65536xf32>, tensor<1x65536xi32>) -> (tensor<1x65536xf32>, tensor<1x65536xi32>)
  %3 = stablehlo.slice %2#0 [0:1, 0:32] : (tensor<1x65536xf32>) -> tensor<1x32xf32>
  %4 = stablehlo.slice %2#1 [0:1, 0:32] : (tensor<1x65536xi32>) -> tensor<1x32xi32>
  %5 = stablehlo.convert %4 : (tensor<1x32xi32>) -> tensor<1x32xi64>
  return %3, %5 : tensor<1x32xf32>, tensor<1x32xi64>
}
