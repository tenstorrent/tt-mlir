// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttnn-trace-hoist-transform --mlir-print-local-scope %s | FileCheck %s

// Verify that creation ops (zeros, ones) are NOT hoisted into the trace
// function since they have the TTCoreCreationOpTrait, and the hoistable ops
// (add, repeat) between them are correctly extracted. Creation ops have no
// func args to merge, so to_layout ops remain in the original function.

module {

  // Trace function: add and repeat hoisted, no creation ops.
  // CHECK-LABEL: func.func private @trace_0_creation_ops
  // CHECK: "ttnn.add"
  // CHECK: "ttnn.repeat"
  // CHECK-NOT: "ttnn.zeros"
  // CHECK-NOT: "ttnn.ones"

  // Capture function: both inputs on system_memory.
  // CHECK-LABEL: func.func private @run_and_capture_trace_0_creation_ops
  // CHECK-SAME: buffer_type<system_memory>
  // CHECK-SAME: buffer_type<system_memory>
  // CHECK: "ttnn.write_tensor"
  // CHECK: "ttnn.write_tensor"
  // CHECK: "ttnn.begin_trace_capture"
  // CHECK: "ttnn.end_trace_capture"

  // CHECK-LABEL: func.func private @execute_trace_0_creation_ops
  // CHECK: "ttnn.execute_trace"

  // Original function: creation ops remain, to_layout converts to system_memory.
  // CHECK-LABEL: func.func @creation_ops
  // CHECK: "ttnn.zeros"
  // CHECK: "ttnn.ones"
  // CHECK: "ttnn.to_layout"
  // CHECK: "ttnn.to_layout"
  // CHECK: "ttnn.capture_or_execute_trace"
  // CHECK-NOT: "ttnn.add"
  // CHECK-NOT: "ttnn.repeat"
  // CHECK: return
  func.func @creation_ops() -> tensor<4x4xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>> attributes {tt.function_type = "forward_device"} {
    %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.zeros"(%0) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<<dram>, <interleaved>>, shape = #ttnn.shape<1x1>}> : (!ttnn.device) -> tensor<1x1xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>
    %2 = "ttnn.ones"(%0) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<<dram>, <interleaved>>, shape = #ttnn.shape<1x1>}> : (!ttnn.device) -> tensor<1x1xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>
    %3 = "ttnn.add"(%1, %2) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<1x1xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>, tensor<1x1xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>) -> tensor<1x1xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>
    %4 = "ttnn.repeat"(%3) <{repeat_dims = #ttnn.shape<4x4>}> : (tensor<1x1xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>) -> tensor<4x4xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>
    return %4 : tensor<4x4xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>
  }
}
