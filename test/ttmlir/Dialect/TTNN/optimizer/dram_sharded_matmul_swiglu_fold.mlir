// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=2 experimental-weight-dtype=bfp_bf8" -o %t %s
// RUN: FileCheck %s --input-file=%t --implicit-check-not='"ttnn.silu"' --implicit-check-not='activation = "silu"'

// SwiGLU: the silu lands on the consuming multiply's operand rather than in the
// matmul kernel, so that the matmul is free to take a DS config.
//
// A DS matmul computes on as many cores as the part has DRAM banks, so an output
// activation runs on that narrow set, while the consumer multiply is free to use
// the whole worker grid. TTNNFusing therefore claims the silu for the multiply
// before the matmul patterns can have it: TTNNBinaryOpInputsActivation is in
// firstPatterns, which reaches a fixpoint before the second set holding
// TTNNMatmulAndLinearWithActivation ever runs. With DRAM sharding on the
// pipeline enables that pattern, threading disable-dram-sharded-matmul into
// ttnn-fusing.
//
// The IR is the eligible-m32 baseline with a silu between the matmul and the
// multiply.
//
// The two implicit-check-not patterns are the failure modes, and they are why
// they are stated that way rather than as CHECK-NOT lines: attributes print in
// alphabetical order, so a surviving `activation` on the matmul appears *before*
// matmul_program_config on the same line, where a CHECK-NOT anchored after the
// program config would not see it. `fused_activation` inside the program config
// is not a third mode -- buildDRAMShardingHint passes a null one deliberately,
// so the op model validates the DS config without an activation.

module attributes {} {
  // CHECK-LABEL: func.func @ds_matmul_swiglu
  // CHECK: "ttnn.matmul"
  // CHECK-SAME: matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_dram_sharded_program_config

  // The fold is positional: the silu'd value is the multiply's LHS here, so the
  // activation lands on operand A. Operand B is covered below.
  // CHECK: "ttnn.multiply"
  // CHECK-SAME: input_tensor_a_activations = [#ttnn.unary_with_param<op_type = silu>]
  func.func @ds_matmul_swiglu(
      %act: tensor<32x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
      %gate_w: tensor<4096x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
      %up: tensor<32x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}) -> tensor<32x4096xbf16> {
    %0 = "ttir.matmul"(%act, %gate_w) : (tensor<32x4096xbf16>, tensor<4096x4096xbf16>) -> tensor<32x4096xbf16>
    %1 = "ttir.silu"(%0) : (tensor<32x4096xbf16>) -> tensor<32x4096xbf16>
    %2 = "ttir.multiply"(%1, %up) : (tensor<32x4096xbf16>, tensor<32x4096xbf16>) -> tensor<32x4096xbf16>
    return %2 : tensor<32x4096xbf16>
  }

  // Same graph with the multiply's operands the other way round. Nothing
  // normalizes the silu'd value onto operand A, so it folds onto operand B --
  // which the flatbuffer, runtime, EmitC and EmitPy paths all carry.
  // CHECK-LABEL: func.func @ds_matmul_swiglu_rhs
  // CHECK: "ttnn.matmul"
  // CHECK-SAME: matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_dram_sharded_program_config

  // CHECK: "ttnn.multiply"
  // CHECK-SAME: input_tensor_b_activations = [#ttnn.unary_with_param<op_type = silu>]
  func.func @ds_matmul_swiglu_rhs(
      %act: tensor<32x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
      %gate_w: tensor<4096x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
      %up: tensor<32x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}) -> tensor<32x4096xbf16> {
    %0 = "ttir.matmul"(%act, %gate_w) : (tensor<32x4096xbf16>, tensor<4096x4096xbf16>) -> tensor<32x4096xbf16>
    %1 = "ttir.silu"(%0) : (tensor<32x4096xbf16>) -> tensor<32x4096xbf16>
    %2 = "ttir.multiply"(%up, %1) : (tensor<32x4096xbf16>, tensor<32x4096xbf16>) -> tensor<32x4096xbf16>
    return %2 : tensor<32x4096xbf16>
  }
}
