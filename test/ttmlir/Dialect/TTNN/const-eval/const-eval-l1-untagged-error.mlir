// RUN: not ttmlir-opt --ttcore-register-device --const-eval-hoist-transform %s 2>&1 | FileCheck %s

// An L1-resident op that satisfies the const-eval criteria (operands trace to
// constant/parameter args) but lacks `ttnn.const_eval_allowed` must fail the
// pass — its L1 reservation would otherwise be invisible to L1 budget
// accounting.

#l1 = #ttnn.buffer_type<l1>

#l1_sharded = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>, core_ranges = <[#ttnn.core_range<(0, 0), (0, 0)>]>>

module {
  func.func @untagged_l1(
      %arg0: tensor<32x32xbf16, #l1_sharded> {ttcore.argument_type = #ttcore.argument_type<input>},
      %arg1: tensor<32x32xbf16, #l1_sharded> {ttcore.argument_type = #ttcore.argument_type<parameter>},
      %arg2: tensor<32x32xbf16, #l1_sharded> {ttcore.argument_type = #ttcore.argument_type<parameter>}
  ) -> tensor<32x32xbf16, #l1_sharded> {
    // CHECK: error: {{.*}}const-eval candidate{{.*}}not tagged with 'ttnn.const_eval_allowed'
    %untagged = "ttnn.add"(%arg1, %arg2) : (tensor<32x32xbf16, #l1_sharded>, tensor<32x32xbf16, #l1_sharded>) -> tensor<32x32xbf16, #l1_sharded>
    %result = "ttnn.add"(%arg0, %untagged) : (tensor<32x32xbf16, #l1_sharded>, tensor<32x32xbf16, #l1_sharded>) -> tensor<32x32xbf16, #l1_sharded>
    return %result : tensor<32x32xbf16, #l1_sharded>
  }
}
