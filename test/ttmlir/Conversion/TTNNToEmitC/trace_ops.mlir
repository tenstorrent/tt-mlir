// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttir-to-ttnn-common-pipeline="enable-const-eval=false enable-trace=true" -o %t.mlir %s
// RUN: ttmlir-opt --ttnn-common-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: FileCheck %s --input-file=%t2.mlir

// --- globals for multi_output ---
// CHECK:       emitc.global @is_first_call_trace_1_multi_output
// CHECK:       emitc.global @trace_id_trace_1_multi_output
// CHECK:       emitc.global @global_input_0_trace_1_multi_output
// CHECK:       emitc.global @global_input_1_trace_1_multi_output
// CHECK:       emitc.global @global_output_0_trace_1_multi_output
// CHECK:       emitc.global @global_output_1_trace_1_multi_output

// --- globals for single_add ---
// CHECK:       emitc.global @is_first_call_trace_0_single_add
// CHECK:       emitc.global @trace_id_trace_0_single_add
// CHECK:       emitc.global @global_input_0_trace_0_single_add
// CHECK:       emitc.global @global_input_1_trace_0_single_add
// CHECK:       emitc.global @global_output_0_trace_0_single_add

// ============================================================================
// Test 1: Single-output trace
// ============================================================================

// CHECK-LABEL: func.func private @trace_0_single_add
// CHECK:       emitc.call_opaque "ttnn::add"

// --- run_and_capture_trace (write_tensor, begin/end trace, execute) ---
// CHECK-LABEL: func.func private @run_and_capture_trace_0_single_add
// CHECK:       emitc.call_opaque "tt::tt_metal::copy_to_device"
// CHECK:       emitc.call_opaque "ttnn::operations::trace::begin_trace_capture"
// CHECK:       emitc.call_opaque "ttnn::operations::trace::end_trace_capture"
// CHECK:       emitc.call_opaque "ttnn::operations::trace::execute_trace"

// --- execute_trace ---
// CHECK-LABEL: func.func private @execute_trace_0_single_add
// CHECK:       emitc.call_opaque "ttnn::operations::trace::execute_trace"

// --- capture_or_execute wrapper ---
// CHECK-LABEL: emitc.func @capture_or_execute_trace_0_single_add
// CHECK:       if
// CHECK:         call_opaque "run_and_capture_trace_0_single_add"
// CHECK:         call_opaque "::std::get"
// CHECK:       } else {
// CHECK:         call_opaque "execute_trace_0_single_add"
// CHECK:       call_opaque "::std::make_tuple"
// CHECK:       return

// --- top-level forward function ---
// CHECK-LABEL: func.func @single_add
// CHECK:       emitc.call_opaque "capture_or_execute_trace_0_single_add"

func.func @single_add(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}) -> tensor<32x32xbf16> {
  %1 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  return %1 : tensor<32x32xbf16>
}

// ============================================================================
// Test 2: Multi-output trace
// ============================================================================

// --- capture_or_execute wrapper ---
// CHECK-LABEL: emitc.func @capture_or_execute_trace_1_multi_output
// CHECK:       if
// CHECK:         call_opaque "run_and_capture_trace_1_multi_output"
// CHECK:         call_opaque "::std::get"
// CHECK:       } else {
// CHECK:         call_opaque "execute_trace_1_multi_output"
// CHECK:       call_opaque "::std::make_tuple"
// CHECK:       return

// --- top-level forward function ---
// CHECK-LABEL: func.func @multi_output
// CHECK:       emitc.call_opaque "capture_or_execute_trace_1_multi_output"
// CHECK:       emitc.call_opaque "::std::get"
// CHECK:       emitc.call_opaque "::std::get"

func.func @multi_output(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}) -> (tensor<32x32xbf16>, tensor<32x32xbf16>) {
  %0 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  %1 = "ttir.multiply"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
  return %0, %1 : tensor<32x32xbf16>, tensor<32x32xbf16>
}
