// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t.mlir %s
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="mock-system-desc-arch=blackhole" -o %t2.mlir %t.mlir
// RUN: FileCheck %s --input-file=%t2.mlir --implicit-check-not="ttnn.indexer_score"

// End-to-end decomposition path: even on Blackhole, a batch size > 1 is not
// supported by the typed ttnn.indexer_score op.
// So the ttcore.composite "indexer_score" is replaced by its
// decomposition of ttnn primitives.

module @indexer_score {
  func.func public @indexer_score(%q: tensor<4x8x32x128xbf16>, %k: tensor<4x1x32x128xbf16>, %w: tensor<4x8x32x1xbf16>) -> tensor<4x1x32x32xbf16> {
    // The primitive ops may be split across the main function and a hoisted
    // const-eval function (the causal mask depends only on shapes), so match
    // them anywhere in the module rather than scoping to a single function.
    // CHECK-DAG: "ttnn.matmul"
    // CHECK-DAG: "ttnn.relu"
    // CHECK-DAG: "ttnn.multiply"
    // CHECK-DAG: "ttnn.sum"
    // CHECK-DAG: "ttnn.arange"
    // CHECK-DAG: "ttnn.ge"
    // CHECK-DAG: "ttnn.where"
    %0 = stablehlo.custom_call @tt.indexer_score(%q, %k, %w) {api_version = 0 : i32} : (tensor<4x8x32x128xbf16>, tensor<4x1x32x128xbf16>, tensor<4x8x32x1xbf16>) -> tensor<4x1x32x32xbf16>
    return %0 : tensor<4x1x32x32xbf16>
  }
}
