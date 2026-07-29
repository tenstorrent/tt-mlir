// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t.mlir %s
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="mock-system-desc-arch=blackhole" -o %t2.mlir %t.mlir
// RUN: FileCheck %s --input-file=%t2.mlir --implicit-check-not="ttnn.sparse_sdpa"

// End-to-end decomposition path: even on Blackhole, a batch size > 1 is not
// supported by the typed ttnn.sparse_sdpa op, so the ttcore.composite
// "sparse_sdpa" is replaced by its decomposition of ttnn primitives.

module @sparse_sdpa {
  func.func public @sparse_sdpa(%q: tensor<4x32x32x64xbf16>, %kv: tensor<4x1x64x64xbf16>, %idx: tensor<4x1x32x32xui32>) -> tensor<4x32x32x32xbf16> {
    // The primitive ops may be split across the main function and a hoisted
    // const-eval function (the shape-only constants get hoisted), so
    // match them anywhere in the module rather than scoping to a single
    // function.
    // CHECK-DAG: "ttnn.matmul"
    // CHECK-DAG: "ttnn.multiply"
    // The sparsity mask is a scatter-accumulate into a [B, S, T] hit-count
    // buffer. It must NOT be a one-hot [B, S, TOPK, T] compare-and-reduce
    // (arange + eq + sum over the slot axis): that needs O(S * TOPK * T) memory,
    // which is 1.07e9 elements at S = T = TOPK = 1024 and does not fit.
    // The upper-bound test canonicalizes into a second "ttnn.gt" with swapped
    // operands, so only the lower bound shows up as "ttnn.ge".
    // CHECK-DAG: "ttnn.ge"
    // CHECK-DAG: "ttnn.scatter"
    // CHECK-DAG: "ttnn.gt"
    // CHECK-DAG: "ttnn.where"
    // CHECK-DAG: "ttnn.softmax"
    %0 = stablehlo.custom_call @tt.sparse_sdpa(%q, %kv, %idx) {api_version = 0 : i32, mhlo.frontend_attributes = {v_dim = "32", k_chunk_size = "32"}} : (tensor<4x32x32x64xbf16>, tensor<4x1x64x64xbf16>, tensor<4x1x32x32xui32>) -> tensor<4x32x32x32xbf16>
    return %0 : tensor<4x32x32x32xbf16>
  }
}
