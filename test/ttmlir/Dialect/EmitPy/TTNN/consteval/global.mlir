// RUN: ttmlir-opt --ttir-to-emitpy-pipeline="system-desc-path=%system_desc_path% enable-const-eval=true" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

// Verify that const-eval caching operations are generated correctly when
// LoadCachedOp is converted from TTNN to EmitPy.

module {
  // CHECK: emitpy.global @_CONST_EVAL_CACHE = #emitpy.opaque<"{i: None for i in range(1)}">

  // CHECK-LABEL: func.func private @forward_const_eval_0

  // CHECK-LABEL: func.func @forward
  func.func @forward(%arg0: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}, %arg1: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg2: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg3: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<32x32xbf16> {
    // CHECK: %{{.*}} = emitpy.global_statement @_CONST_EVAL_CACHE : !emitpy.dict<index, !emitpy.opaque<"[ttnn.Tensor]">>
    // CHECK: %{{.*}} = emitpy.get_value_for_dict_key "_CONST_EVAL_CACHE" %{{.*}}[0 : index] : (!emitpy.dict<index, !emitpy.opaque<"[ttnn.Tensor]">>) -> !emitpy.opaque<"[ttnn.Tensor]">
    // CHECK: emitpy.set_value_for_dict_key "_CONST_EVAL_CACHE" %{{.*}}[0 : index] = %{{.*}} : (!emitpy.dict<index, !emitpy.opaque<"[ttnn.Tensor]">>) -> !emitpy.opaque<"[ttnn.Tensor]">
    %1 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %3 = "ttir.add"(%arg1, %arg2) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %5 = "ttir.add"(%arg2, %arg3) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %7 = "ttir.subtract"(%3, %5) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %9 = "ttir.multiply"(%1, %7) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %9 : tensor<32x32xbf16>
  }
}
