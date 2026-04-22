// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=2 experimental-kv-cache-dtype=bfp_bf8" %s | FileCheck %s
//
// Regression test: the greedy optimizer must preserve bfp_bf8 for fill_cache
// operands when --experimental-kv-cache-dtype=bfp_bf8 is active.
//
// Before the constraint-sink fix, ValidationAndFallback downgraded both
// fill_cache operands from bfp_bf8 to bf16 because:
//   1. The optimizer (skipping fill_cache as a no-result op) freely assigned
//      block_sharded L1 bfp_bf8 to the input tensor.
//   2. That sharding violated fill_cache's shard_width == padded_width
//      constraint (shard width 64 != padded width 128).
//   3. bfp_bf8 was not in the ValidationAndFallback fallback dtype list,
//      so it could not recover bfp_bf8 interleaved and fell back to bf16.
//
// After the fix, fill_cache participates as a constraint sink in the optimizer:
// input layout combinations are validated, the optimizer picks height_sharded
// bfp_bf8 (shard width == padded width), and no dtype fallback is needed.

// CHECK-LABEL: func.func @main
// Both fill_cache operands must be bfp_bf8 — no bf16 downcast.
// CHECK: "ttnn.fill_cache"
// CHECK-SAME: bfp_bf8
// CHECK-SAME: bfp_bf8

module attributes {} {
  func.func @main(
      %cache: tensor<32x8x128x128xbf16>,
      %input: tensor<1x8x18x128xbf16>
  ) -> tensor<32x8x128x128xbf16> {
    %result = "ttir.fill_cache"(%cache, %input) <{batch_offset = 0 : i32}>
        : (tensor<32x8x128x128xbf16>, tensor<1x8x18x128xbf16>)
        -> tensor<32x8x128x128xbf16>
    return %result : tensor<32x8x128x128xbf16>
  }
}
