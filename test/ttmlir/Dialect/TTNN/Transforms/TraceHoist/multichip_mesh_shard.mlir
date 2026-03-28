// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path% mesh-shape=1,2" --ttnn-trace-hoist-transform --mlir-print-local-scope %s | FileCheck %s

// Verify that multichip workloads with mesh_shard ops are correctly handled.
// mesh_shard ops should NOT be hoisted. Regular inputs through mesh_shard are
// converted to system_memory. After mergeToLayoutOpsWithFuncArgs, mesh_shard
// output types also become system_memory.

module {

  // Trace function: only add, no mesh_shard.
  // CHECK-LABEL: func.func private @trace_0_main
  // CHECK: "ttnn.add"
  // CHECK-NOT: "ttnn.mesh_shard"

  // Capture function: both inputs on system_memory.
  // CHECK-LABEL: func.func private @run_and_capture_trace_0_main
  // CHECK-SAME: buffer_type<system_memory>
  // CHECK-SAME: buffer_type<system_memory>
  // CHECK: "ttnn.write_tensor"
  // CHECK: "ttnn.write_tensor"
  // CHECK: "ttnn.begin_trace_capture"
  // CHECK: "ttnn.end_trace_capture"

  // CHECK-LABEL: func.func private @execute_trace_0_main
  // CHECK: "ttnn.execute_trace"

  // Original function: args merged to system_memory, mesh_shard ops remain.
  // CHECK-LABEL: func.func @main
  // CHECK-SAME: buffer_type<system_memory>
  // CHECK-SAME: buffer_type<system_memory>
  // CHECK: "ttnn.mesh_shard"
  // CHECK: "ttnn.mesh_shard"
  // CHECK: "ttnn.capture_or_execute_trace"
  // CHECK: "ttnn.mesh_shard"
  // CHECK-NOT: "ttnn.add"
  // CHECK: return
  func.func @main(%arg0: tensor<10x128x512xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 128 + d1, d2), <1x1>, memref<40x16x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<5x128x512xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>}, %arg1: tensor<10x128x512xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 128 + d1, d2), <1x1>, memref<40x16x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.local_shape = #ttcore<local_shape local_shape = tensor<5x128x512xbf16>>, ttcore.shard_status = #ttcore.shard_status<presharded>}) -> (tensor<10x128x512xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 128 + d1, d2), <1x1>, memref<40x16x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>> {ttcore.local_shape = #ttcore<local_shape local_shape = tensor<10x128x512xbf16>>, ttcore.shard_status = #ttcore.shard_status<unsharded>}) attributes {tt.function_type = "forward_device"} {
    %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x2>}> : () -> !ttnn.device
    %1 = "ttnn.mesh_shard"(%arg0, %0) <{shard_dims = array<i64: -1, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 2, 1, 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<10x128x512xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 128 + d1, d2), <1x1>, memref<40x16x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>, !ttnn.device) -> tensor<5x128x512xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 128 + d1, d2), <1x1>, memref<20x16x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>
    %2 = "ttnn.mesh_shard"(%arg1, %0) <{shard_dims = array<i64: -1, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 2, 1, 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<10x128x512xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 128 + d1, d2), <1x1>, memref<40x16x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>, !ttnn.device) -> tensor<5x128x512xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 128 + d1, d2), <1x1>, memref<20x16x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>
    %3 = "ttnn.add"(%2, %1) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<5x128x512xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 128 + d1, d2), <1x1>, memref<20x16x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>, tensor<5x128x512xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 128 + d1, d2), <1x1>, memref<20x16x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>) -> tensor<5x128x512xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 128 + d1, d2), <1x1>, memref<20x16x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>
    %4 = "ttnn.mesh_shard"(%3, %0) <{shard_dims = array<i64: -1, 0>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 2, 1, 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<5x128x512xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 128 + d1, d2), <1x1>, memref<20x16x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>, !ttnn.device) -> tensor<10x128x512xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 128 + d1, d2), <1x1>, memref<40x16x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>
    return %4 : tensor<10x128x512xbf16, #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 128 + d1, d2), <1x1>, memref<40x16x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>
  }
}
