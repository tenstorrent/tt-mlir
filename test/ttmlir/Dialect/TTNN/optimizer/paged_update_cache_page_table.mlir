// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=2 experimental-kv-cache-dtype=bfp_bf8" -o %t %s -mlir-print-local-scope
// RUN: FileCheck %s --input-file=%t

// Regression: opt-2 with a bfp8 KV cache. 4-operand paged_update_cache -- the
// optimizer's layout decision for the page_table operand (operand 3) must stay
// compatible with the ttnn op. Both indices are computed (argmax over a runtime
// input) so the optimizer may relayout them.

// CHECK: "ttnn.paged_update_cache"
func.func @main(
    %cache: tensor<7x4x32x128xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.kv_cache},
    %input: tensor<1x1x4x128xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
    %idx_scores: tensor<1x4xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
    %pt_scores: tensor<1x1x4xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}
) -> tensor<7x4x32x128xbf16> {
    %update_index = "ttir.argmax"(%idx_scores) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<1x4xbf16>) -> tensor<1xi32>
    %page_table = "ttir.argmax"(%pt_scores) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<1x1x4xbf16>) -> tensor<1x1xi32>
    %0 = "ttir.paged_update_cache"(%cache, %input, %update_index, %page_table) <{share_cache = false}> : (tensor<7x4x32x128xbf16>, tensor<1x1x4x128xbf16>, tensor<1xi32>, tensor<1x1xi32>) -> tensor<7x4x32x128xbf16>
    return %0 : tensor<7x4x32x128xbf16>
}
