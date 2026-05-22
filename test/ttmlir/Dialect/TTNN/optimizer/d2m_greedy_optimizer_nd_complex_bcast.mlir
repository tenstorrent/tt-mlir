// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=2 enable-create-d2m-subgraphs=true" --mlir-print-local-scope -o %t %s
// RUN: FileCheck %s --input-file=%t

// End-to-end regression: a TTIR graph with a *complex* implicit broadcast
// (an outer broadcast dim that the TTNN layout collapses next to non-trivial
// inner dims, so the per-shard physical dim is neither equal to the output's
// nor 1) used to crash the full ttir-to-ttnn-backend-pipeline inside
// `D2MNamedElementwiseRewriter::buildPhysicalImplicitBcastIndexingMaps`
// with `Assertion `inShard[d] == 1' failed`.
//
// The fix materializes the cross-tile broadcast pattern via a
// `d2m.view_layout` with a `mod`/`floordiv` remapping so the d2m.generic
// itself only sees simple identity / constant-0 indexing maps, and lets the
// pipeline run all the way down to `ttnn.generic` compiled kernels.

module {
  // 5D RoPE-style cos chain: (q * cos) + (q_rot * sin). Both `cos` and `sin`
  // are broadcast on the outer batch dim from 1 -> 32, and their tile shard
  // dim (16) is neither equal to the output's tile shard dim (4096) nor 1
  // after the TTNN layout collapse.
  func.func @nd5_rope_chain_d8(
      %q     : tensor<32x16x8x32x1xf32>,
      %cos   : tensor<1x16x1x32x1xf32>,
      %q_rot : tensor<32x16x8x32x1xf32>,
      %sin   : tensor<1x16x1x32x1xf32>) -> tensor<32x16x8x32x1xf32> {
    %0 = "ttir.multiply"(%q, %cos) : (tensor<32x16x8x32x1xf32>, tensor<1x16x1x32x1xf32>) -> tensor<32x16x8x32x1xf32>
    %1 = "ttir.multiply"(%q_rot, %sin) : (tensor<32x16x8x32x1xf32>, tensor<1x16x1x32x1xf32>) -> tensor<32x16x8x32x1xf32>
    %2 = "ttir.add"(%0, %1) : (tensor<32x16x8x32x1xf32>, tensor<32x16x8x32x1xf32>) -> tensor<32x16x8x32x1xf32>
    return %2 : tensor<32x16x8x32x1xf32>
  }
}

// CHECK: func.func @nd5_rope_chain_d8
// The broadcasted-then-fused chain should reach ttnn.generic kernels rather
// than being left as native ttnn.add / ttnn.multiply.
// CHECK-NOT: "ttnn.add"
// CHECK-NOT: "ttnn.multiply"
// CHECK: "ttnn.generic"
