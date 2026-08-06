// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-pipeline -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

// tt.zeros_buffer has no operands, so the only thing that can shard its
// result is the built-in rule registered in RegisterCustomShardingRule.cpp
// (one free pass-through factor per result dimension). Without that rule
// Shardy warns "sharding rule is not defined for target" and leaves the result
// replicated: a caller's sharding constraint then becomes a reshard *after* a
// full global-shaped allocation instead of constraining the allocation itself.
//
// The discriminating check is the local shape of the custom_call inside the
// manual_computation body: it must be one shard's worth, not the global shape.

sdy.mesh @mesh = <["x"=2]>

// Shard dim 0: 8 -> 4.
// CHECK-LABEL: func.func @create_sharded_cache
// CHECK: sdy.manual_computation()
// CHECK-SAME: in_shardings=[]
// CHECK-SAME: out_shardings=[<@mesh, [{"x"}, {}]>]
// CHECK: stablehlo.custom_call @tt.zeros_buffer
// CHECK-SAME: () -> tensor<4x16xf32>
func.func @create_sharded_cache() -> tensor<8x16xf32> {
  %0 = stablehlo.custom_call @tt.zeros_buffer() {has_side_effect = true} : () -> tensor<8x16xf32>
  %1 = sdy.sharding_constraint %0 <@mesh, [{"x"}, {}]> : tensor<8x16xf32>
  return %1 : tensor<8x16xf32>
}

// The real KV cache shape, sharded on the head axis (dim 1): 16 -> 8.
// CHECK-LABEL: func.func @create_sharded_kv_cache
// CHECK: sdy.manual_computation()
// CHECK-SAME: out_shardings=[<@mesh, [{}, {"x"}, {}, {}]>]
// CHECK: stablehlo.custom_call @tt.zeros_buffer
// CHECK-SAME: () -> tensor<64x8x32x128xbf16>
func.func @create_sharded_kv_cache() -> tensor<64x16x32x128xbf16> {
  %0 = stablehlo.custom_call @tt.zeros_buffer() {has_side_effect = true} : () -> tensor<64x16x32x128xbf16>
  %1 = sdy.sharding_constraint %0 <@mesh, [{}, {"x"}, {}, {}]> : tensor<64x16x32x128xbf16>
  return %1 : tensor<64x16x32x128xbf16>
}
