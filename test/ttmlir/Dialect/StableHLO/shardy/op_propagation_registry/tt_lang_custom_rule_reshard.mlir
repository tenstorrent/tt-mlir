// REQUIRES: stablehlo
// RUN: ttmlir-opt --register-custom-sharding-rule --register-user-sharding-rule \
// RUN:   --sdy-user-priority-propagate --insert-explicit-reshards \
// RUN:   -split-input-file -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

// A user-provided rule must make operand alignment work the same way a built-in
// C++ rule (e.g. RMSNorm's) does: when propagation gives the result a sharding
// that an operand does not have, InsertExplicitReshards has to insert an
// sdy.reshard on that operand.
//
// That only happens if the rule is still reachable *after* propagation, which
// requires two things working together:
//
//   1. The rule is non-custom. Shardy's removeShardingRules strips non-custom
//      rules at the end of propagation, and InsertExplicitReshards bails out on
//      rules that are custom (b/434668939). So a rule kept custom would survive
//      the strip but then be skipped, and never produce a reshard.
//
//   2. Once stripped, the rule can be recreated. getOrCreateShardingRule falls
//      back to ShardingRuleOpInterface, which re-parses the frontend attribute
//      for targets with no built-in rule.

sdy.mesh @mesh = <["x"=2]>

// %arg0 is sharded on dim 0 and %arg1 is explicitly replicated. The rule makes
// dim 0 a shared factor, so the result picks up "x" from %arg0 and %arg1 has to
// be resharded to match.
//
// CHECK-LABEL: func.func @operand_needs_reshard
// CHECK: %[[RESHARD:.*]] = sdy.reshard %arg1 <@mesh, [{"x"}, {}]>
// CHECK: stablehlo.custom_call @tt.tt_lang_op(%arg0, %[[RESHARD]])
// CHECK-SAME: sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x", ?}, {?}]>]>
func.func @operand_needs_reshard(
    %arg0: tensor<512x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
    %arg1: tensor<512x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>})
    -> tensor<512x64xf32> {
  %0 = stablehlo.custom_call @tt.tt_lang_op(%arg0, %arg1) {
    api_version = 0 : i32,
    mhlo.frontend_attributes = {
      arg_roles = "in,in,out",
      "xla.sdy.custom_sharding_rule" = "#sdy.op_sharding_rule<([i, j], [i, j]) -> ([i, j]) {i=512, j=64}>"
    }
  } : (tensor<512x64xf32>, tensor<512x64xf32>) -> tensor<512x64xf32>
  return %0 : tensor<512x64xf32>
}

// -----

sdy.mesh @mesh = <["x"=2]>

// Same graph, but the user wrote the rule as `custom`. The promote pass
// normalizes that away, so the reshard is still inserted. Without the
// normalization the rule would stay on the op and be skipped, silently leaving
// %arg1 unaligned with the sharded result.
//
// CHECK-LABEL: func.func @custom_flag_is_normalized_away
// CHECK: sdy.reshard %arg1 <@mesh, [{"x"}, {}]>
// The normalized rule is non-custom, so propagation strips it.
// CHECK-NOT: sdy.sharding_rule =
func.func @custom_flag_is_normalized_away(
    %arg0: tensor<512x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
    %arg1: tensor<512x64xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>})
    -> tensor<512x64xf32> {
  %0 = stablehlo.custom_call @tt.tt_lang_op(%arg0, %arg1) {
    api_version = 0 : i32,
    mhlo.frontend_attributes = {
      arg_roles = "in,in,out",
      "xla.sdy.custom_sharding_rule" = "#sdy.op_sharding_rule<([i, j], [i, j]) -> ([i, j]) {i=512, j=64}, custom>"
    }
  } : (tensor<512x64xf32>, tensor<512x64xf32>) -> tensor<512x64xf32>
  return %0 : tensor<512x64xf32>
}
