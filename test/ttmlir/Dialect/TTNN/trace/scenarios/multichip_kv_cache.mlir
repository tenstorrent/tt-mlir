// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="mesh-shape=1,2 enable-trace=true" --mlir-print-local-scope %s | FileCheck %s

// Verify that KV cache arguments flowing through mesh_shard ops are correctly
// detected and kept on device during trace hoisting (not moved to host).

module {
    // In the capture function, the KV cache arg should remain on DRAM (device),
    // while the regular input should be on system_memory (host).
    // Constants are hoisted as the first arguments.
    //
    // CHECK-LABEL: func.func private @run_and_capture_trace_0_main
    // Constant args come first.
    // CHECK-SAME: %arg0:
    // CHECK-SAME: ttcore.argument_type = #ttcore.argument_type<constant>
    // CHECK-SAME: %arg1:
    // CHECK-SAME: ttcore.argument_type = #ttcore.argument_type<constant>
    // The regular input arg should be on system_memory (host).
    // CHECK-SAME: %arg2: tensor<5x128x512xbf16,
    // CHECK-SAME: buffer_type<system_memory>
    // The KV cache arg should remain on dram (device), marked with kv_cache.
    // CHECK-SAME: %arg3: tensor<1x32x64x512xbf16,
    // CHECK-SAME: buffer_type<dram>
    // CHECK-SAME: ttcore.kv_cache

    // CHECK-LABEL: func.func @main(
    // CHECK: "ttnn.mesh_shard"
    // CHECK: "ttnn.mesh_shard"
    func.func @main(
        %arg0: tensor<10x128x512xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<5x128x512xbf16>>} loc("p0.2"),
        %cache: tensor<2x32x64x512xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.shard_status = #ttcore.shard_status<presharded>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x32x64x512xbf16>>} loc("p1.3")
    ) -> (tensor<10x128x512xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<10x128x512xbf16>>},
          tensor<2x32x64x512xbf16> {ttcore.shard_status = #ttcore.shard_status<presharded>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x32x64x512xbf16>>}) {
        %0 = "ttir.mesh_shard"(%arg0) <{shard_dims = array<i64: -1, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 2, 1, 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<10x128x512xbf16>) -> tensor<5x128x512xbf16>
        %sharded_cache = "ttir.mesh_shard"(%cache) <{shard_dims = array<i64: -1, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 2, 1, 1, 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<2x32x64x512xbf16>) -> tensor<1x32x64x512xbf16>
        %idx = "ttir.constant"() <{value = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
        %update_input = "ttir.constant"() <{value = dense<1.0> : tensor<1x32x1x512xbf16>}> : () -> tensor<1x32x1x512xbf16>
        %updated_cache = "ttir.update_cache"(%sharded_cache, %update_input, %idx) <{batch_offset = 0 : i32}> : (tensor<1x32x64x512xbf16>, tensor<1x32x1x512xbf16>, tensor<1xi32>) -> tensor<1x32x64x512xbf16>
        %1 = "ttir.add"(%0, %0) : (tensor<5x128x512xbf16>, tensor<5x128x512xbf16>) -> tensor<5x128x512xbf16>
        %2 = "ttir.mesh_shard"(%1) <{shard_dims = array<i64: -1, 0>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 2, 1, 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<5x128x512xbf16>) -> tensor<10x128x512xbf16>
        %3 = "ttir.mesh_shard"(%updated_cache) <{shard_dims = array<i64: -1, 0>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 2, 1, 1, 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<1x32x64x512xbf16>) -> tensor<2x32x64x512xbf16>
        return %2, %3 : tensor<10x128x512xbf16>, tensor<2x32x64x512xbf16>
    }
}
