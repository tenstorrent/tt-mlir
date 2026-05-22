// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

// Tests that tt.sharded_topk custom_call decomposes into:
//   ttir.topk  (local topk on each shard)
//   ttir.all_gather x2  (tiny values + indices across devices)
//   ttir.constant  (compile-time shard offset)
//   ttir.add  (local → global vocab indices)

module attributes {} {

  // 4-shard case: batch=32, vocab_per_shard=32768, k=32, num_shards=4
  // Represents a 4-chip setup with 128k vocab (128k/4 = 32k per chip).
  // CHECK-LABEL: func.func @sharded_topk_4shards
  func.func @sharded_topk_4shards(
      %arg0: tensor<32x32768xbf16>) -> (tensor<32x128xbf16>, tensor<32x128xi32>) {

    // Local topk on the 32k shard → 32 candidates per device.
    // CHECK: "ttir.topk"(%arg0)
    // CHECK-SAME: k = 32
    // CHECK-SAME: -> (tensor<32x32xbf16>, tensor<32x32xi32>)

    // All-gather values and indices (tiny: 32 per device → 128 total).
    // CHECK: "ttir.all_gather"
    // CHECK-SAME: all_gather_dim = 1
    // CHECK-SAME: tensor<32x128xbf16>
    // CHECK: "ttir.all_gather"
    // CHECK-SAME: all_gather_dim = 1
    // CHECK-SAME: tensor<32x128xi32>

    // Compile-time shard offset constant:
    // [0..0, 32768..32768, 65536..65536, 98304..98304] (32 copies each)
    // CHECK: "ttir.constant"
    // CHECK-SAME: tensor<32x128xi32>

    // Add offset to convert local → global vocab indices.
    // CHECK: "ttir.add"
    // CHECK-SAME: tensor<32x128xi32>

    %0:2 = stablehlo.custom_call @tt.sharded_topk(%arg0) {
        api_version = 0 : i32,
        mhlo.frontend_attributes = {k = "32", num_shards = "4"}
    } : (tensor<32x32768xbf16>) -> (tensor<32x128xbf16>, tensor<32x128xi32>)
    return %0#0, %0#1 : tensor<32x128xbf16>, tensor<32x128xi32>
  }

  // 2-shard case: batch=32, vocab_per_shard=65536, k=32, num_shards=2
  // Represents a 2-chip setup with 128k vocab.
  // CHECK-LABEL: func.func @sharded_topk_2shards
  func.func @sharded_topk_2shards(
      %arg0: tensor<32x65536xbf16>) -> (tensor<32x64xbf16>, tensor<32x64xi32>) {

    // CHECK: "ttir.topk"(%arg0)
    // CHECK-SAME: k = 32
    // CHECK-SAME: -> (tensor<32x32xbf16>, tensor<32x32xi32>)

    // CHECK: "ttir.all_gather"
    // CHECK-SAME: tensor<32x64xbf16>
    // CHECK: "ttir.all_gather"
    // CHECK-SAME: tensor<32x64xi32>

    // CHECK: "ttir.constant"
    // CHECK-SAME: tensor<32x64xi32>

    // CHECK: "ttir.add"
    // CHECK-SAME: tensor<32x64xi32>

    %0:2 = stablehlo.custom_call @tt.sharded_topk(%arg0) {
        api_version = 0 : i32,
        mhlo.frontend_attributes = {k = "32", num_shards = "2"}
    } : (tensor<32x65536xbf16>) -> (tensor<32x64xbf16>, tensor<32x64xi32>)
    return %0#0, %0#1 : tensor<32x64xbf16>, tensor<32x64xi32>
  }

}
