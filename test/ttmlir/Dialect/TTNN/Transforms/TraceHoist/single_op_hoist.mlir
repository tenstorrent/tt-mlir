// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttnn-trace-hoist-transform --mlir-print-local-scope %s | FileCheck %s

// Verify that a single hoistable op (add) is extracted into trace/capture/execute
// functions and replaced with capture_or_execute_trace in the original function.
// After mergeToLayoutOpsWithFuncArgs, the original function args become
// system_memory and no to_layout ops remain.

module {

  // The trace function should contain the hoisted add op on dram.
  // CHECK-LABEL: func.func private @trace_0_single_add
  // CHECK-SAME: buffer_type<dram>
  // CHECK-SAME: buffer_type<dram>
  // CHECK: "ttnn.add"

  // The capture function should have both inputs as system_memory.
  // CHECK-LABEL: func.func private @run_and_capture_trace_0_single_add
  // CHECK-SAME: buffer_type<system_memory>
  // CHECK-SAME: buffer_type<system_memory>
  // CHECK: "ttnn.empty"
  // CHECK: "ttnn.empty"
  // CHECK: "ttnn.write_tensor"
  // CHECK: "ttnn.write_tensor"
  // CHECK: "ttnn.begin_trace_capture"
  // CHECK: "ttnn.end_trace_capture"

  // CHECK-LABEL: func.func private @execute_trace_0_single_add
  // CHECK: "ttnn.execute_trace"

  // The original function: args merged to system_memory, no to_layout.
  // CHECK-LABEL: func.func @single_add
  // CHECK-SAME: buffer_type<system_memory>
  // CHECK-SAME: buffer_type<system_memory>
  // CHECK: "ttnn.capture_or_execute_trace"
  // CHECK-NOT: "ttnn.add"
  // CHECK: return
  func.func @single_add(%arg0: tensor<32x32xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>, %arg1: tensor<32x32xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>) -> tensor<32x32xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>> attributes {tt.function_type = "forward_device"} {
    %0 = "ttnn.add"(%arg0, %arg1) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<32x32xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>, tensor<32x32xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>) -> tensor<32x32xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>
    return %0 : tensor<32x32xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>
  }
}
