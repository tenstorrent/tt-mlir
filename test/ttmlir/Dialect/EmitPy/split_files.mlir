// RUN: ttmlir-opt --ttir-to-emitpy-pipeline %s | FileCheck %s

// This test validates the TTIR to EmitPy pipeline with split-files enabled
// (default). The const-eval computation is split into a separate consteval file op.

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
// CHECK:   emitpy.import import "ttnn"
// CHECK:   emitpy.import import "utils"
// CHECK:   emitpy.file "main" {
// CHECK:     emitpy.import from "consteval" import "cpu_hoisted_const_eval_{{.*}}", "consteval_forward"
// CHECK:     func.func private @cpu_hoisted_const_eval_{{.*}}
// CHECK:     func.func private @consteval_forward
// CHECK:     emitpy.global @_cached_forward = #emitpy.opaque<"{}">
// CHECK:     func.func @forward(
// CHECK:       emitpy.global_statement @_cached_forward
// CHECK:       call @consteval_forward({{.*}})
// CHECK:       emitpy.call_opaque "ttnn.add"
// CHECK:       emitpy.call_opaque "ttnn.add"
// CHECK:   }
// CHECK:   emitpy.file "consteval" {
// CHECK:     func.func @cpu_hoisted_const_eval_{{.*}}
// CHECK:     func.func private @forward_const_eval_0(
// CHECK:       call @cpu_hoisted_const_eval_{{.*}}
// CHECK:     func.func @consteval_forward(
// CHECK:       emitpy.if "not {}"
// CHECK:         emitpy.call_opaque "forward_const_eval_0"
// CHECK:   }
// CHECK: }
