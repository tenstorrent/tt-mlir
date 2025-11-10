// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-const-eval=true" -o %t.mlir %s
// RUN: ttmlir-opt --ttnn-backend-to-emitpy-pipeline -o %t2.mlir %t.mlir
// RUN: FileCheck %s --input-file=%t2.mlir

// Verify that all 4 GlobalOp-related operations are generated correctly when
// LoadCachedOp is converted from TTNN to EmitPy.

module {
  func.func @forward(%arg0: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}, %arg1: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg2: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg3: tensor<32x32xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<32x32xbf16> {
    %0 = ttir.empty() : tensor<32x32xbf16>
    %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %2 = ttir.empty() : tensor<32x32xbf16>
    %3 = "ttir.add"(%arg1, %arg2, %2) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %4 = ttir.empty() : tensor<32x32xbf16>
    %5 = "ttir.add"(%arg2, %arg3, %4) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %6 = ttir.empty() : tensor<32x32xbf16>
    %7 = "ttir.subtract"(%3, %5, %6) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %8 = ttir.empty() : tensor<32x32xbf16>
    %9 = "ttir.multiply"(%1, %7, %8) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %9 : tensor<32x32xbf16>
  }
}

// CHECK-LABEL: func.func @forward_const_eval_0

// CHECK: emitpy.global @g_cached_result_forward_const_eval_0 : #emitpy.opaque<"[]">
// CHECK-LABEL: func.func @forward
// CHECK: %{{.*}} = emitpy.global_statement @g_cached_result_forward_const_eval_0 : !emitpy.opaque<"[ttnn.Tensor]">
// CHECK: %{{.*}} = emitpy.get_global @g_cached_result_forward_const_eval_0 : !emitpy.opaque<"[ttnn.Tensor]">
// CHECK: %{{.*}} = "emitpy.assign_global"(%{{.*}}) <{name = @g_cached_result_forward_const_eval_0}> : (!emitpy.opaque<"[ttnn.Tensor]">) -> !emitpy.opaque<"[ttnn.Tensor]">
