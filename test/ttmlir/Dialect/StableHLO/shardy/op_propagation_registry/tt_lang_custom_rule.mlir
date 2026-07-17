// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-pipeline -o %t.mlir %s 2>%t.err
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: FileCheck %s --check-prefix=ERR --input-file=%t.err

// End-to-end coverage for user-provided `xla.sdy.custom_sharding_rule` rules on a
// `tt.tt_lang_op` custom_call, exercised through the full stablehlo pipeline:
//   * @with_rule      -- a valid rule is parsed into a typed
//                        `sdy.op_sharding_rule`, honored by Shardy propagation,
//                        and reflected in the local shapes produced by
//                        UpdateGlobalToLocalShapes.
//   * @without_rule   -- no rule: Shardy has no built-in rule for tt.tt_lang_op,
//                        so the result is left replicated.
//
// A malformed or semantically invalid rule is a hard error and is covered
// separately in tt_lang_custom_rule_invalid.mlir.
//
// `tt.tt_lang_op` has no built-in sharding rule, so the custom rule is the only
// thing that can shard the result: the sharded (4x16) result in @with_rule is
// what makes the test discriminating.

sdy.mesh @mesh = <["x"=2]>

// With the rule, dim 0 is a shared factor ([i, j], [i, j]) -> ([i, j]): the
// operands' "x" sharding on dim 0 propagates to the result, so every tensor
// inside the manual_computation body is local-shaped 4x16 (8 / 2).
//
// CHECK-LABEL: func.func @with_rule
// The rule propagates the operand sharding to the result...
// CHECK: sdy.manual_computation(%arg0, %arg1)
// CHECK-SAME: in_shardings=[<@mesh, [{"x"}, {}]>, <@mesh, [{"x"}, {}]>]
// CHECK-SAME: out_shardings=[<@mesh, [{"x"}, {}]>]
// ...the string rule was parsed into a typed, reprinted sdy.op_sharding_rule...
// CHECK: stablehlo.custom_call @tt.tt_lang_op
// CHECK-SAME: sdy.sharding_rule = #sdy.op_sharding_rule<([i, j], [i, j])->([i, j]) {i=8, j=16}, custom>
// ...and UpdateGlobalToLocalShapes halves dim 0 (8 -> 4) on operands AND result.
// CHECK-SAME: (tensor<4x16xf32>, tensor<4x16xf32>) -> tensor<4x16xf32>
// CHECK: sdy.return %{{.*}} : tensor<4x16xf32>
func.func @with_rule(
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
    %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
    -> tensor<8x16xf32> {
  %0 = stablehlo.custom_call @tt.tt_lang_op(%arg0, %arg1) {
    api_version = 0 : i32,
    mhlo.frontend_attributes = {
      kernel_id = "pkg.add::v1",
      version_tag = "1.0",
      arg_roles = "in,out",
      "xla.sdy.custom_sharding_rule" = "#sdy.op_sharding_rule<([i, j], [i, j]) -> ([i, j]) {i=8, j=16}, custom>"
    }
  } : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// Without a rule there is nothing to relate operands and result: Shardy leaves
// the result unconstrained (replicated), so UpdateGlobalToLocalShapes keeps the
// result at its full 8x16 shape even though the operands are local 4x16.
//
// CHECK-LABEL: func.func @without_rule
// Result sharding is empty (replicated), not {"x"}.
// CHECK: out_shardings=[<@mesh, [{}, {}]>]
// Operands are local 4x16 but the result stays full 8x16.
// CHECK: stablehlo.custom_call @tt.tt_lang_op
// CHECK-SAME: (tensor<4x16xf32>, tensor<4x16xf32>) -> tensor<8x16xf32>
// CHECK: sdy.return %{{.*}} : tensor<8x16xf32>
//
// With no rule and no built-in rule for tt.tt_lang_op, Shardy warns loudly.
// ERR-DAG: warning: StableHLO CustomCallOp sharding rule is not defined for target 'tt.tt_lang_op'
func.func @without_rule(
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
    %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
    -> tensor<8x16xf32> {
  %0 = stablehlo.custom_call @tt.tt_lang_op(%arg0, %arg1) {
    api_version = 0 : i32,
    mhlo.frontend_attributes = {
      kernel_id = "pkg.add::v1",
      version_tag = "1.0",
      arg_roles = "in,out"
    }
  } : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// An empty (or whitespace-only) rule string is treated exactly like no rule: it
// is NOT a hard parse error, and the op falls back to replication just like
// @without_rule (result stays full 8x16, operands local 4x16). This guards the
// empty-string fallback in the promote pass.
//
// CHECK-LABEL: func.func @empty_rule
// CHECK: out_shardings=[<@mesh, [{}, {}]>]
// CHECK: stablehlo.custom_call @tt.tt_lang_op
// CHECK-SAME: (tensor<4x16xf32>, tensor<4x16xf32>) -> tensor<8x16xf32>
// CHECK: sdy.return %{{.*}} : tensor<8x16xf32>
func.func @empty_rule(
    %arg0: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>},
    %arg1: tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>})
    -> tensor<8x16xf32> {
  %0 = stablehlo.custom_call @tt.tt_lang_op(%arg0, %arg1) {
    api_version = 0 : i32,
    mhlo.frontend_attributes = {
      kernel_id = "pkg.add::v1",
      version_tag = "1.0",
      arg_roles = "in,out",
      "xla.sdy.custom_sharding_rule" = ""
    }
  } : (tensor<8x16xf32>, tensor<8x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}
