// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path% mesh-shape=1,2" --ttnn-trace-hoist-transform --mlir-print-local-scope %s | FileCheck %s

// Verify that KV cache arguments flowing through mesh_shard ops are correctly
// detected and kept on device during trace hoisting (not moved to host).
// Also tests that load_cached results (including L1 height_sharded) are kept
// on device.

module {

  func.func private @main_const_eval_0() -> tensor<1x1x32x512xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1, (d0, d1) -> (0, d0 floordiv 8, d0 mod 8)>, memref<1x16x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <height_sharded>>> attributes {tt.function_type = "const_eval"} {
    %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x2>}> : () -> !ttnn.device
    %1 = "ttnn.full"(%0) <{dtype = #ttcore.supportedDataTypes<bf16>, fill_value = 1.000000e+00 : f32, layout = #ttnn.layout<tile>, shape = #ttnn.shape<1x32x1x512>}> : (!ttnn.device) -> tensor<1x32x1x512xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<32x16x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>
    %2 = "ttnn.permute"(%1) <{permutation = array<i64: 0, 2, 1, 3>}> : (tensor<1x32x1x512xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<32x16x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>) -> tensor<1x1x32x512xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x16x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>
    %3 = "ttnn.to_layout"(%2) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<<l1>, <height_sharded>, #ttnn.shard_spec<<[#ttnn.core_range<(0,0), (0,0)>]>, <32x512>, <row_major>>>}> : (tensor<1x1x32x512xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x16x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>) -> tensor<1x1x32x512xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1, (d0, d1) -> (0, d0 floordiv 8, d0 mod 8)>, memref<1x16x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <height_sharded>>>
    return %3 : tensor<1x1x32x512xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1, (d0, d1) -> (0, d0 floordiv 8, d0 mod 8)>, memref<1x16x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <height_sharded>>>
  }

  func.func private @main_const_eval_1() -> tensor<1xsi32, #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1xsi32, #ttnn.buffer_type<dram>>, <interleaved>>> attributes {tt.function_type = "const_eval"} {
    %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x2>}> : () -> !ttnn.device
    %1 = "ttnn.full"(%0) <{dtype = #ttcore.supportedDataTypes<si32>, fill_value = 0 : i32, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<<dram>, <interleaved>>, shape = #ttnn.shape<1>}> : (!ttnn.device) -> tensor<1xsi32, #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1xsi32, #ttnn.buffer_type<dram>>, <interleaved>>>
    return %1 : tensor<1xsi32, #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1xsi32, #ttnn.buffer_type<dram>>, <interleaved>>>
  }

  // Trace function: paged_update_cache and add hoisted.
  // CHECK-LABEL: func.func private @trace_0_main
  // CHECK-SAME: {ttcore.argument_type = #ttcore.argument_type<constant>}
  // CHECK-SAME: {ttcore.argument_type = #ttcore.argument_type<constant>}
  // CHECK-SAME: ttcore.shard_status
  // CHECK-SAME: ttcore.kv_cache
  // CHECK: "ttnn.paged_update_cache"
  // CHECK: "ttnn.add"

  // Capture function:
  // - load_cached L1 result: constant, stays on l1
  // - load_cached DRAM result: constant, stays on dram
  // - regular mesh_shard output: system_memory
  // - KV cache mesh_shard output: stays on dram
  // CHECK-LABEL: func.func private @run_and_capture_trace_0_main
  // CHECK-SAME: %arg0: tensor<1x1x32x512xbf16,
  // CHECK-SAME: buffer_type<l1>
  // CHECK-SAME: {ttcore.argument_type = #ttcore.argument_type<constant>}
  // CHECK-SAME: %arg1: tensor<1xsi32,
  // CHECK-SAME: buffer_type<dram>
  // CHECK-SAME: {ttcore.argument_type = #ttcore.argument_type<constant>}
  // CHECK-SAME: %arg2: tensor<5x128x512xbf16,
  // CHECK-SAME: buffer_type<system_memory>
  // CHECK-SAME: %arg3: tensor<1x32x64x512xbf16,
  // CHECK-SAME: buffer_type<dram>
  // CHECK-SAME: ttcore.kv_cache
  // Only one write_tensor (for the regular input).
  // CHECK: "ttnn.write_tensor"
  // CHECK-NOT: "ttnn.write_tensor"
  // CHECK: "ttnn.begin_trace_capture"
  // CHECK: "ttnn.end_trace_capture"

  // CHECK-LABEL: func.func private @execute_trace_0_main
  // CHECK: "ttnn.execute_trace"

  // Original function: arg0 merged to system_memory, arg1 (kv_cache) stays dram.
  // CHECK-LABEL: func.func @main
  // CHECK-SAME: %arg0: tensor<10x128x512xbf16,
  // CHECK-SAME: buffer_type<system_memory>
  // CHECK-SAME: %arg1: tensor<2x32x64x512xbf16,
  // CHECK-SAME: buffer_type<dram>
  // CHECK-SAME: ttcore.kv_cache
  // CHECK: ttcore.load_cached
  // CHECK: ttcore.load_cached
  // CHECK: "ttnn.mesh_shard"
  // CHECK: "ttnn.mesh_shard"
  // CHECK: "ttnn.capture_or_execute_trace"
  // CHECK: "ttnn.mesh_shard"
  // CHECK: "ttnn.mesh_shard"
  // CHECK: return
  func.func @main(%arg0: tensor<10x128x512xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 128 + d1, d2), <1x1>, memref<40x16x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<5x128x512xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>}, %arg1: tensor<2x32x64x512xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 2048 + d1 * 64 + d2, d3), <1x1>, memref<128x16x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.kv_cache, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x32x64x512xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>}) -> (tensor<10x128x512xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 128 + d1, d2), <1x1>, memref<40x16x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>> {ttcore.local_shape = #ttcore<local_shape local_shape = tensor<10x128x512xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>}, tensor<2x32x64x512xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 2048 + d1 * 64 + d2, d3), <1x1>, memref<128x16x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>> {ttcore.local_shape = #ttcore<local_shape local_shape = tensor<1x32x64x512xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>}) attributes {tt.function_type = "forward_device"} {
    %0 = ttcore.load_cached(@main_const_eval_0, []) : () -> tensor<1x1x32x512xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1, (d0, d1) -> (0, d0 floordiv 8, d0 mod 8)>, memref<1x16x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <height_sharded>>>
    %1 = ttcore.load_cached(@main_const_eval_1, []) : () -> tensor<1xsi32, #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1xsi32, #ttnn.buffer_type<dram>>, <interleaved>>>
    %2 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x2>}> : () -> !ttnn.device
    %3 = "ttnn.mesh_shard"(%arg0, %2) <{shard_dims = array<i64: -1, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 2, 1, 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<10x128x512xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 128 + d1, d2), <1x1>, memref<40x16x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>, !ttnn.device) -> tensor<5x128x512xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 128 + d1, d2), <1x1>, memref<20x16x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>
    %4 = "ttnn.mesh_shard"(%arg1, %2) <{shard_dims = array<i64: -1, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 2, 1, 1, 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<2x32x64x512xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 2048 + d1 * 64 + d2, d3), <1x1>, memref<128x16x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>, !ttnn.device) -> tensor<1x32x64x512xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 2048 + d1 * 64 + d2, d3), <1x1>, memref<64x16x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>
    "ttnn.paged_update_cache"(%4, %0, %1) <{share_cache = false}> : (tensor<1x32x64x512xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 2048 + d1 * 64 + d2, d3), <1x1>, memref<64x16x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>, tensor<1x1x32x512xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1, (d0, d1) -> (0, d0 floordiv 8, d0 mod 8)>, memref<1x16x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <height_sharded>>>, tensor<1xsi32, #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1xsi32, #ttnn.buffer_type<dram>>, <interleaved>>>) -> ()
    %5 = "ttnn.add"(%3, %3) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<5x128x512xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 128 + d1, d2), <1x1>, memref<20x16x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>, tensor<5x128x512xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 128 + d1, d2), <1x1>, memref<20x16x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>) -> tensor<5x128x512xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 128 + d1, d2), <1x1>, memref<20x16x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>
    %6 = "ttnn.mesh_shard"(%5, %2) <{shard_dims = array<i64: -1, 0>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 2, 1, 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<5x128x512xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 128 + d1, d2), <1x1>, memref<20x16x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>, !ttnn.device) -> tensor<10x128x512xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 128 + d1, d2), <1x1>, memref<40x16x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>
    %7 = "ttnn.mesh_shard"(%4, %2) <{shard_dims = array<i64: -1, 0>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 2, 1, 1, 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<1x32x64x512xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 2048 + d1 * 64 + d2, d3), <1x1>, memref<64x16x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>, !ttnn.device) -> tensor<2x32x64x512xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 2048 + d1 * 64 + d2, d3), <1x1>, memref<128x16x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>
    return %6, %7 : tensor<10x128x512xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 128 + d1, d2), <1x1>, memref<40x16x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>, tensor<2x32x64x512xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 2048 + d1 * 64 + d2, d3), <1x1>, memref<128x16x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>
  }
}
