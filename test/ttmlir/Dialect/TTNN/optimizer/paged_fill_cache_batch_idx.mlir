// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=2 experimental-kv-cache-dtype=bfp_bf8" -o %t %s -mlir-print-local-scope
// RUN: FileCheck %s --input-file=%t

// Regression: opt-2 with a bfp8 KV cache. The optimizer's layout decision for
// the paged_fill_cache batch_idx operand (operand 3) must stay compatible with
// the ttnn op. batch_idx is computed (argmax over a runtime input) so the
// optimizer may relayout it.

// CHECK: "ttnn.paged_fill_cache"
func.func @main(
    %cache: tensor<128x12x32x256xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.kv_cache},
    %input: tensor<1x12x65x256xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
    %page_table: tensor<8x16xi32> {ttcore.argument_type = #ttcore.argument_type<input>},
    %bidx_scores: tensor<1x4xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}
) -> tensor<128x12x32x256xbf16> {
    %batch_idx = "ttir.argmax"(%bidx_scores) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<1x4xbf16>) -> tensor<1xi32>
    %0 = "ttir.paged_fill_cache"(%cache, %input, %page_table, %batch_idx) : (tensor<128x12x32x256xbf16>, tensor<1x12x65x256xbf16>, tensor<8x16xi32>, tensor<1xi32>) -> tensor<128x12x32x256xbf16>
    return %0 : tensor<128x12x32x256xbf16>
}
