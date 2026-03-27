// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttnn-trace-hoist-transform --mlir-print-local-scope %s | FileCheck %s

// Verify that load_cached results (consteval) are kept on device during trace
// hoisting, while regular inputs are converted to system_memory.

module {

  func.func private @matmul_with_multiply_const_eval_0(%arg0: tensor<64x32xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x32xbf16, #ttnn.buffer_type<system_memory>>>>, %arg1: tensor<32x64xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x64xbf16, #ttnn.buffer_type<system_memory>>>>) -> tensor<64x64xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>> attributes {tt.function_type = "const_eval"} {
    %0 = "ttnn.to_layout"(%arg1) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<<dram>, <interleaved>>}> : (tensor<32x64xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x64xbf16, #ttnn.buffer_type<system_memory>>>>) -> tensor<32x64xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>
    %1 = "ttnn.to_layout"(%arg0) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<<dram>, <interleaved>>}> : (tensor<64x32xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x32xbf16, #ttnn.buffer_type<system_memory>>>>) -> tensor<64x32xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>
    %2 = "ttnn.matmul"(%1, %0) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = false}> : (tensor<64x32xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>, tensor<32x64xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>) -> tensor<64x64xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>
    return %2 : tensor<64x64xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>
  }

  // Trace function: to_layout (from input) + multiply hoisted.
  // The load_cached constant arg stays on dram.
  // CHECK-LABEL: func.func private @trace_0_matmul_with_multiply
  // CHECK-SAME: {ttcore.argument_type = #ttcore.argument_type<input>}
  // CHECK-SAME: {ttcore.argument_type = #ttcore.argument_type<constant>}
  // CHECK: "ttnn.to_layout"
  // CHECK: "ttnn.multiply"

  // Capture function: regular input on system_memory, constant on dram.
  // CHECK-LABEL: func.func private @run_and_capture_trace_0_matmul_with_multiply
  // CHECK-SAME: %arg0: tensor<64x64xbf16,
  // CHECK-SAME: buffer_type<system_memory>
  // CHECK-SAME: {ttcore.argument_type = #ttcore.argument_type<input>}
  // CHECK-SAME: %arg1: tensor<64x64xbf16,
  // CHECK-SAME: buffer_type<dram>
  // CHECK-SAME: {ttcore.argument_type = #ttcore.argument_type<constant>}
  // Only one write_tensor (for the regular input, not the constant).
  // CHECK: "ttnn.write_tensor"
  // CHECK-NOT: "ttnn.write_tensor"
  // CHECK: "ttnn.begin_trace_capture"
  // CHECK: "ttnn.end_trace_capture"

  // CHECK-LABEL: func.func private @execute_trace_0_matmul_with_multiply
  // CHECK: "ttnn.execute_trace"

  // Original function: arg2 (input) merged to system_memory.
  // CHECK-LABEL: func.func @matmul_with_multiply
  // CHECK: ttcore.load_cached
  // CHECK: "ttnn.capture_or_execute_trace"
  // CHECK-NOT: "ttnn.multiply"
  // CHECK: return
  func.func @matmul_with_multiply(%arg0: tensor<64x32xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x32xbf16, #ttnn.buffer_type<system_memory>>>> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg1: tensor<32x64xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x64xbf16, #ttnn.buffer_type<system_memory>>>> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg2: tensor<64x64xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x64xbf16, #ttnn.buffer_type<dram>>, <interleaved>>> {ttcore.argument_type = #ttcore.argument_type<input>}) -> tensor<64x64xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>> attributes {tt.function_type = "forward_device"} {
    %0 = ttcore.load_cached(@matmul_with_multiply_const_eval_0, [%arg0, %arg1]) : (tensor<64x32xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x32xbf16, #ttnn.buffer_type<system_memory>>>>, tensor<32x64xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x64xbf16, #ttnn.buffer_type<system_memory>>>>) -> tensor<64x64xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>
    %1 = "ttnn.to_layout"(%arg2) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<<dram>, <interleaved>>}> : (tensor<64x64xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x64xbf16, #ttnn.buffer_type<dram>>, <interleaved>>>) -> tensor<64x64xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>
    %2 = "ttnn.multiply"(%0, %1) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<64x64xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>, tensor<64x64xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>) -> tensor<64x64xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>
    return %2 : tensor<64x64xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>
  }
}
