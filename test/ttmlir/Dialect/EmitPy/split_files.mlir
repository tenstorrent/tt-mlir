// RUN: ttmlir-opt --ttir-to-emitpy-pipeline %s | FileCheck %s

// This test validates the CodegenSplitFiles pass through the TTIR to EmitPy pipeline.
// The test verifies:
// - TTIR input with const-eval operations
// - Pipeline creates separate emitpy.file ops for "main" and "consteval"
// - Correct imports and function placement in each file

module {
  func.func @forward(%arg0: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
                     %arg1: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
                     %arg2: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<32x32xbf16> {
    // This add uses input, so it stays in main
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    // This add only uses parameters/constants, so it gets hoisted to const-eval
    %1 = "ttir.add"(%arg1, %arg2) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    // Final add combines both
    %2 = "ttir.add"(%0, %1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %2 : tensor<32x32xbf16>
  }
}

// CHECK: module {
// CHECK:   emitpy.file "main" {
// CHECK:     emitpy.import import "ttnn"
// CHECK:     emitpy.import import "utils"
// CHECK:     emitpy.import from "consteval" import "consteval_forward"
// CHECK:     func.func @forward(
// CHECK:       emitpy.call_opaque "consteval_forward"
// CHECK:       emitpy.call_opaque "ttnn.add"
// CHECK:   }
// CHECK:   emitpy.file "consteval" {
// CHECK:     emitpy.import import "ttnn"
// CHECK:     emitpy.import import "utils"
// CHECK:     emitpy.global @_CONST_EVAL_CACHE
// CHECK:     func.func @cpu_hoisted_const_eval_{{.*}}
// CHECK:       ttnn.add
// CHECK:     func.func private @forward_const_eval_0
// CHECK:       call @cpu_hoisted_const_eval_{{.*}}
// CHECK:     func.func @consteval_forward(
// CHECK:   }
// CHECK: }
