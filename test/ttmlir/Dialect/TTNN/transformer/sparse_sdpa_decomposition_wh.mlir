// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t.mlir %s
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="mock-system-desc-arch=wormhole_b0" -o %t2.mlir %t.mlir
// RUN: FileCheck %s --input-file=%t2.mlir --implicit-check-not="ttnn.sparse_sdpa"

// End-to-end decomposition path: on a non-Blackhole target (Wormhole), the
// ttcore.composite "sparse_sdpa" is replaced by its decomposition of ttnn
// primitives.

module @sparse_sdpa {
  func.func public @sparse_sdpa(%q: tensor<1x32x32x64xbf16>, %kv: tensor<1x1x64x64xbf16>, %idx: tensor<1x1x32x32xui32>) -> tensor<1x32x32x32xbf16> {
    // The primitive ops may be split across the main function and a hoisted
    // const-eval function (the key-position arange depends only on shapes), so
    // match them anywhere in the module rather than scoping to a single
    // function.
    // CHECK-DAG: "ttnn.matmul"
    // CHECK-DAG: "ttnn.multiply"
    // CHECK-DAG: "ttnn.arange"
    // CHECK-DAG: "ttnn.eq"
    // CHECK-DAG: "ttnn.sum"
    // CHECK-DAG: "ttnn.gt"
    // CHECK-DAG: "ttnn.where"
    // CHECK-DAG: "ttnn.softmax"
    %0 = stablehlo.custom_call @tt.sparse_sdpa(%q, %kv, %idx) {api_version = 0 : i32, mhlo.frontend_attributes = {v_dim = "32", k_chunk_size = "32"}} : (tensor<1x32x32x64xbf16>, tensor<1x1x64x64xbf16>, tensor<1x1x32x32xui32>) -> tensor<1x32x32x32xbf16>
    return %0 : tensor<1x32x32x32xbf16>
  }
}
