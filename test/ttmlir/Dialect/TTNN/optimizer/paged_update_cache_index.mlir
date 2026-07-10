// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=2 experimental-kv-cache-dtype=bfp_bf8" -o %t %s -mlir-print-local-scope
// RUN: FileCheck %s --input-file=%t

// Regression: at opt-2 with a bfp8 KV cache, the optimizer's layout decisions
// for the paged_update_cache operands must stay compatible with the ttnn op.
// The index is computed (argmax over a runtime input), so it is neither a
// row-major-seeded integer argument nor const-evaluable -- the optimizer is
// free to relayout it.

// CHECK: "ttnn.paged_update_cache"
func.func @main(
    %cache: tensor<1x8x16x128xbf16> {ttcore.argument_type = #ttcore.argument_type<input>, ttcore.kv_cache},
    %input: tensor<1x8x1x128xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
    %scores: tensor<1x4xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}
) -> tensor<1x8x16x128xbf16> {
    %idx = "ttir.argmax"(%scores) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<1x4xbf16>) -> tensor<1xi32>
    %0 = "ttir.update_cache"(%cache, %input, %idx) <{batch_offset = 0 : i32}> : (tensor<1x8x16x128xbf16>, tensor<1x8x1x128xbf16>, tensor<1xi32>) -> tensor<1x8x16x128xbf16>
    return %0 : tensor<1x8x16x128xbf16>
}
