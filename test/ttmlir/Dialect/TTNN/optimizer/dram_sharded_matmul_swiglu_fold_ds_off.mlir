// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=2 experimental-weight-dtype=bfp_bf8 disable-dram-sharded-matmul=true" -o %t %s
// RUN: FileCheck %s --input-file=%t --implicit-check-not='"ttnn.silu"'

// The DS-off half of dram_sharded_matmul_swiglu_fold.mlir, same IR and same
// pipeline but for the kill switch.
//
// With DRAM sharding off there is no narrow-grid matmul to keep the activation
// away from, so the silu goes where it did before: onto the matmul, via
// TTNNMatmulAndLinearWithActivation. That pattern is registered unconditionally
// and only ever loses the silu to TTNNBinaryOpInputsActivation, which the
// pipeline enables solely when DRAM sharding is on.
//
// This is what pins the ordering. Both patterns can claim the same silu, and
// which one wins is decided by firstPatterns reaching a fixpoint before the
// second set runs -- not by anything in the greedy driver's own ordering. Assert
// both halves or a regression that dropped the split registration would still
// pass the DS-on test.

module attributes {} {
  // The activation ends up inside the program config rather than as an op-level
  // attribute: the non-DS apply path folds it into fused_activation and then
  // removes the attribute, so it is not applied twice (tt-metal #35060).
  // CHECK-LABEL: func.func @ds_matmul_swiglu_ds_off
  // CHECK: "ttnn.matmul"
  // CHECK-SAME: fused_activation = #ttnn.unary_with_param<op_type = silu>
  // CHECK-NOT: dram_sharded_program_config

  // The multiply is left with empty activation lists: with the inputs pattern
  // off, nothing folds onto its operands.
  // CHECK: "ttnn.multiply"
  // CHECK-SAME: input_tensor_a_activations = []
  // CHECK-SAME: input_tensor_b_activations = []
  func.func @ds_matmul_swiglu_ds_off(
      %act: tensor<32x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
      %gate_w: tensor<4096x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
      %up: tensor<32x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}) -> tensor<32x4096xbf16> {
    %0 = "ttir.matmul"(%act, %gate_w) : (tensor<32x4096xbf16>, tensor<4096x4096xbf16>) -> tensor<32x4096xbf16>
    %1 = "ttir.silu"(%0) : (tensor<32x4096xbf16>) -> tensor<32x4096xbf16>
    %2 = "ttir.multiply"(%1, %up) : (tensor<32x4096xbf16>, tensor<32x4096xbf16>) -> tensor<32x4096xbf16>
    return %2 : tensor<32x4096xbf16>
  }
}
