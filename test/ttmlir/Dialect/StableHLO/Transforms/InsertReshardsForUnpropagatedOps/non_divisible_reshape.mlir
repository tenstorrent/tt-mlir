// REQUIRES: stablehlo
// RUN: ttmlir-opt --split-input-file --insert-reshards-for-unpropagated-ops -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

// Shardy can propagate a valid-looking result sharding on a reshape's flat
// dimension while the operand's factored dimension is non-divisible by the
// shard count (e.g. heads=30 split across model=4). The old anyResultHasSharding
// early-out skipped these ops, leaving an unrealizable reshape for
// UpdateGlobalToLocalShapes. The pass now reshards the sharded operand to
// replicated and drops the unrealizable result sharding. See tt-xla#5148.

sdy.mesh @mesh = <["model"=4]>

// 120 is divisible by model=4 (-> 30 per shard) but the factored dim 30 is not,
// so the result sharding cannot localize and the reshape must be replicated.
// CHECK-LABEL: func.func @nondivisible_split
func.func @nondivisible_split(
    %arg0: tensor<120xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"model"}]>})
    -> tensor<30x4xf32> {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{}]> : tensor<120xf32>
  // CHECK: stablehlo.reshape %[[RESHARD]] : (tensor<120xf32>) -> tensor<30x4xf32>
  // CHECK-NOT: sdy.sharding_per_value
  %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"model"}, {}]>]>} : (tensor<120xf32>) -> tensor<30x4xf32>
  return %0 : tensor<30x4xf32>
}

// -----

sdy.mesh @mesh = <["model"=4]>

// Original tt-xla#3643 case: Shardy left the result unannotated while a sharded
// operand shrinks per shard. The per-shard element counts diverge, so the
// operand is still resharded to replicated (behavior preserved/subsumed).
// CHECK-LABEL: func.func @unannotated_result
func.func @unannotated_result(
    %arg0: tensor<120xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"model"}]>})
    -> tensor<30x4xf32> {
  // CHECK: %[[RESHARD:.*]] = sdy.reshard %arg0 <@mesh, [{}]> : tensor<120xf32>
  // CHECK: stablehlo.reshape %[[RESHARD]] : (tensor<120xf32>) -> tensor<30x4xf32>
  %0 = stablehlo.reshape %arg0 : (tensor<120xf32>) -> tensor<30x4xf32>
  return %0 : tensor<30x4xf32>
}

// -----

sdy.mesh @mesh = <["model"=4]>

// Divisible/consistent reshape: operand 4x8 sharded on model (-> 1x8 = 8 elems
// per shard) merges to a flat dim sharded on model (32 -> 8 per shard). The
// per-shard counts match, so the pass leaves the op untouched (no reshard, the
// result sharding is preserved).
// CHECK-LABEL: func.func @divisible_merge_unchanged
func.func @divisible_merge_unchanged(
    %arg0: tensor<4x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"model"}, {}]>})
    -> tensor<32xf32> {
  // CHECK-NOT: sdy.reshard
  // CHECK: stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"model"}]>]>}
  %0 = stablehlo.reshape %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"model"}]>]>} : (tensor<4x8xf32>) -> tensor<32xf32>
  return %0 : tensor<32xf32>
}
