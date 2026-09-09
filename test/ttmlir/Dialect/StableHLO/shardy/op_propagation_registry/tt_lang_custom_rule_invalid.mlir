// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-pipeline -verify-diagnostics=only-expected -split-input-file %s

// An invalid `xla.sdy.custom_sharding_rule` frontend attribute is a hard error, not a
// silent fallback to replication. There are two distinct failure points:
//
//   * A rule string that does not parse into an #sdy.op_sharding_rule is caught
//     by RegisterUserShardingRulePass, which fails before propagation runs
//     (see @not_a_rule / @wrong_attribute_kind).
//
//   * A rule that parses but is semantically inconsistent with the op it is
//     attached to (operand/result arity, per-tensor rank, or factor reuse) is
//     caught by Shardy's own always-on op-attribute verifier: once the pass
//     promotes the rule to the `sdy.sharding_rule` op attribute, MLIR operation
//     verification runs SdyDialect::verifyOperationAttribute ->
//     verifyOpShardingRuleAttr, which rejects it with a clear diagnostic in
//     every build (see @arity_mismatch / @rank_mismatch / @duplicate_factor).
//     These diagnostics are emitted by Shardy, so we match Shardy's wording.
//
// `only-expected` is used because the full pipeline also emits unrelated
// module-level warnings (e.g. missing argument type map / result shard status)
// and the MLIR parser emits its own sub-diagnostic for an unparseable rule; we
// only want to assert on the error under test here.

sdy.mesh @mesh = <["x"=2]>

func.func @not_a_rule(
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
    %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
    -> tensor<8x16xf32> {
  // expected-error @+1 {{failed to parse 'xla.sdy.custom_sharding_rule' frontend attribute as an #sdy.op_sharding_rule}}
  %0 = stablehlo.custom_call @tt.tt_lang_op(%arg0, %arg1) {
    api_version = 0 : i32,
    mhlo.frontend_attributes = {
      arg_roles = "in,out",
      "xla.sdy.custom_sharding_rule" = "not a valid rule"
    }
  } : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// -----

sdy.mesh @mesh = <["x"=2]>

func.func @wrong_attribute_kind(
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
    %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
    -> tensor<8x16xf32> {
  // A syntactically valid attribute that is not an #sdy.op_sharding_rule.
  // expected-error @+1 {{failed to parse 'xla.sdy.custom_sharding_rule' frontend attribute as an #sdy.op_sharding_rule}}
  %0 = stablehlo.custom_call @tt.tt_lang_op(%arg0, %arg1) {
    api_version = 0 : i32,
    mhlo.frontend_attributes = {
      arg_roles = "in,out",
      "xla.sdy.custom_sharding_rule" = "42 : i32"
    }
  } : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// -----

sdy.mesh @mesh = <["x"=2]>

// Op has 2 operands but the rule only maps 1 operand. Caught by Shardy's
// verifyOpShardingRuleAttr once the rule is promoted to the op attribute.
func.func @arity_mismatch(
    %arg0: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
    %arg1: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
    -> tensor<8x4xf32> {
  // expected-error @+1 {{'stablehlo.custom_call' op number of operands and mappings must match: 2 != 1}}
  %0 = stablehlo.custom_call @tt.tt_lang_op(%arg0, %arg1) {
    api_version = 0 : i32,
    mhlo.frontend_attributes = {
      arg_roles = "in,out",
      "xla.sdy.custom_sharding_rule" = "#sdy.op_sharding_rule<([i, j])->([i, j]) {i=8, j=4}, custom>"
    }
  } : (tensor<8x4xf32>, tensor<8x4xf32>) -> tensor<8x4xf32>
  return %0 : tensor<8x4xf32>
}

// -----

sdy.mesh @mesh = <["x"=2]>

// Op operand is rank 2 but the rule's operand mapping is rank 3.
func.func @rank_mismatch(
    %arg0: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
    -> tensor<8x4xf32> {
  // expected-error @+1 {{'stablehlo.custom_call' op operand - mapping rank must match: 3 != 2}}
  %0 = stablehlo.custom_call @tt.tt_lang_op(%arg0) {
    api_version = 0 : i32,
    mhlo.frontend_attributes = {
      arg_roles = "in,out",
      "xla.sdy.custom_sharding_rule" = "#sdy.op_sharding_rule<([i, j, k])->([i, j]) {i=8, j=4, k=1}, custom>"
    }
  } : (tensor<8x4xf32>) -> tensor<8x4xf32>
  return %0 : tensor<8x4xf32>
}

// -----

sdy.mesh @mesh = <["x"=2]>

// Factor `i` is mapped to both dim 0 and dim 1 of the same tensor.
func.func @duplicate_factor(
    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
    -> tensor<8x8xf32> {
  // expected-error @+1 {{'stablehlo.custom_call' op operand - cannot reuse factors for the same tensor value}}
  %0 = stablehlo.custom_call @tt.tt_lang_op(%arg0) {
    api_version = 0 : i32,
    mhlo.frontend_attributes = {
      arg_roles = "in,out",
      "xla.sdy.custom_sharding_rule" = "#sdy.op_sharding_rule<([i, i])->([i, i]) {i=8}, custom>"
    }
  } : (tensor<8x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}
