// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-to-ttnn-common-pipeline="enable-const-eval=false enable-trace=true" -o %t.mlir %s
// RUN: ttmlir-opt --ttnn-common-to-emitpy-pipeline -o %t2.mlir %t.mlir
// RUN: FileCheck %s --input-file=%t2.mlir

// --- globals for multi_output ---
// CHECK:       emitpy.global @is_first_call_trace_1_multi_output
// CHECK:       emitpy.global @trace_id_trace_1_multi_output
// CHECK:       emitpy.global @global_input_0_trace_1_multi_output
// CHECK:       emitpy.global @global_input_1_trace_1_multi_output
// CHECK:       emitpy.global @global_output_0_trace_1_multi_output
// CHECK:       emitpy.global @global_output_1_trace_1_multi_output

// --- capture_or_execute wrapper for multi_output ---
// CHECK-LABEL: func.func private @capture_or_execute_trace_1_multi_output
// CHECK:       emitpy.if
// CHECK:         emitpy.call_opaque "run_and_capture_trace_1_multi_output"
// CHECK:       } else {
// CHECK:         emitpy.call_opaque "execute_trace_1_multi_output"
// CHECK:       return

// --- globals for single_add ---
// CHECK:       emitpy.global @is_first_call_trace_0_single_add
// CHECK:       emitpy.global @trace_id_trace_0_single_add
// CHECK:       emitpy.global @global_input_0_trace_0_single_add
// CHECK:       emitpy.global @global_input_1_trace_0_single_add
// CHECK:       emitpy.global @global_output_0_trace_0_single_add

// --- capture_or_execute wrapper for single_add ---
// CHECK-LABEL: func.func private @capture_or_execute_trace_0_single_add
// CHECK:       emitpy.if
// CHECK:         emitpy.call_opaque "run_and_capture_trace_0_single_add"
// CHECK:       } else {
// CHECK:         emitpy.call_opaque "execute_trace_0_single_add"
// CHECK:       return

// --- trace function for single_add ---
// CHECK-LABEL: func.func private @trace_0_single_add
// CHECK:       emitpy.call_opaque "ttnn.add"

// --- run_and_capture_trace (write_tensor, begin/end trace, execute) for single_add ---
// CHECK-LABEL: func.func private @run_and_capture_trace_0_single_add
// CHECK:       emitpy.call_opaque "ttnn.copy_host_to_device_tensor"
// CHECK:       emitpy.call_opaque "ttnn.begin_trace_capture"
// CHECK:       emitpy.call_opaque "ttnn.end_trace_capture"
// CHECK:       emitpy.call_opaque "ttnn.execute_trace"

// --- execute_trace for single_add ---
// CHECK-LABEL: func.func private @execute_trace_0_single_add
// CHECK:       emitpy.call_opaque "ttnn.execute_trace"

// --- top-level single_add forward function ---
// CHECK-LABEL: func.func @single_add
// CHECK:       emitpy.call_opaque "capture_or_execute_trace_0_single_add"

func.func @single_add(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}) -> tensor<32x32xbf16> {
  %1 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  return %1 : tensor<32x32xbf16>
}

// --- trace function for multi_output ---
// CHECK-LABEL: func.func private @trace_1_multi_output
// CHECK:       emitpy.call_opaque "ttnn.add"
// CHECK:       emitpy.call_opaque "ttnn.multiply"

// --- run_and_capture_trace (write_tensor, begin/end trace, execute) for multi_output ---
// CHECK-LABEL: func.func private @run_and_capture_trace_1_multi_output
// CHECK:       emitpy.call_opaque "ttnn.copy_host_to_device_tensor"
// CHECK:       emitpy.call_opaque "ttnn.begin_trace_capture"
// CHECK:       emitpy.call_opaque "ttnn.end_trace_capture"
// CHECK:       emitpy.call_opaque "ttnn.execute_trace"

// --- execute_trace for multi_output ---
// CHECK-LABEL: func.func private @execute_trace_1_multi_output
// CHECK:       emitpy.call_opaque "ttnn.execute_trace"

// --- top-level multi_output forward function ---
// CHECK-LABEL: func.func @multi_output
// CHECK:       emitpy.call_opaque "capture_or_execute_trace_1_multi_output"

func.func @multi_output(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}) -> (tensor<32x32xbf16>, tensor<32x32xbf16>) {
  %0 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  %1 = "ttir.multiply"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  return %0, %1 : tensor<32x32xbf16>, tensor<32x32xbf16>
}
