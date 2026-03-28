// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttnn-trace-hoist-transform --mlir-print-local-scope %s | FileCheck %s

// Verify that multiple hoistable ops (matmul + multiply) are extracted into
// trace/capture/execute functions. After mergeToLayoutOpsWithFuncArgs, the
// original function args become system_memory.

module {

  // Trace function: both matmul and multiply hoisted.
  // CHECK-LABEL: func.func private @trace_0_matmul_with_multiply
  // CHECK: "ttnn.matmul"
  // CHECK: "ttnn.multiply"

  // Capture function: all three inputs as system_memory.
  // CHECK-LABEL: func.func private @run_and_capture_trace_0_matmul_with_multiply
  // CHECK-SAME: buffer_type<system_memory>
  // CHECK-SAME: buffer_type<system_memory>
  // CHECK-SAME: buffer_type<system_memory>
  // CHECK: "ttnn.empty"
  // CHECK: "ttnn.empty"
  // CHECK: "ttnn.empty"
  // CHECK: "ttnn.write_tensor"
  // CHECK: "ttnn.write_tensor"
  // CHECK: "ttnn.write_tensor"
  // CHECK: "ttnn.begin_trace_capture"
  // CHECK: "ttnn.end_trace_capture"

  // CHECK-LABEL: func.func private @execute_trace_0_matmul_with_multiply
  // CHECK: "ttnn.execute_trace"

  // Original function: args merged to system_memory, no to_layout.
  // CHECK-LABEL: func.func @matmul_with_multiply
  // CHECK-SAME: buffer_type<system_memory>
  // CHECK-SAME: buffer_type<system_memory>
  // CHECK-SAME: buffer_type<system_memory>
  // CHECK: "ttnn.capture_or_execute_trace"
  // CHECK-NOT: "ttnn.matmul"
  // CHECK-NOT: "ttnn.multiply"
  // CHECK: return
  func.func @matmul_with_multiply(%arg0: tensor<64x32xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>, %arg1: tensor<32x64xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>, %arg2: tensor<64x64xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>) -> tensor<64x64xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>> attributes {tt.function_type = "forward_device"} {
    %0 = "ttnn.matmul"(%arg0, %arg1) <{compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>, transpose_a = false, transpose_b = false}> : (tensor<64x32xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>, tensor<32x64xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>) -> tensor<64x64xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>
    %1 = "ttnn.multiply"(%0, %arg2) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<64x64xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>, tensor<64x64xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>) -> tensor<64x64xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>
    return %1 : tensor<64x64xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>
  }
}
