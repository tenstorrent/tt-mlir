// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=2 experimental-weight-dtype=bfp_bf8 enable-dram-sharded-matmul=true" -o %t %s
// RUN: FileCheck %s --input-file=%t --implicit-check-not='"ttnn.silu"' --implicit-check-not='activation = "silu"'

// Shapes the DS gate accepts; each gets the DS program config with per_core_m = 1.
// The implicit-check-nots catch a silu that survived outside the multiply.

module attributes {} {
  // Baseline decode projection.
  // CHECK-LABEL: func.func @ds_matmul_m32
  // CHECK: matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_dram_sharded_program_config
  // CHECK-SAME: per_core_m = 1
  func.func @ds_matmul_m32(
      %act: tensor<32x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
      %weight: tensor<4096x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
      %other: tensor<32x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}) -> tensor<32x4096xbf16> {
    %0 = "ttir.matmul"(%act, %weight) : (tensor<32x4096xbf16>, tensor<4096x4096xbf16>) -> tensor<32x4096xbf16>
    %1 = "ttir.multiply"(%0, %other) : (tensor<32x4096xbf16>, tensor<32x4096xbf16>) -> tensor<32x4096xbf16>
    return %1 : tensor<32x4096xbf16>
  }

  // A sub-tile batch pads up to one tile row.
  // CHECK-LABEL: func.func @ds_matmul_batch1
  // CHECK: matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_dram_sharded_program_config
  // CHECK-SAME: per_core_m = 1
  func.func @ds_matmul_batch1(
      %act: tensor<1x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
      %weight: tensor<4096x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
      %other: tensor<1x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}) -> tensor<1x4096xbf16> {
    %0 = "ttir.matmul"(%act, %weight) : (tensor<1x4096xbf16>, tensor<4096x4096xbf16>) -> tensor<1x4096xbf16>
    %1 = "ttir.multiply"(%0, %other) : (tensor<1x4096xbf16>, tensor<1x4096xbf16>) -> tensor<1x4096xbf16>
    return %1 : tensor<1x4096xbf16>
  }

  // A [1, 1, K, N] weight is the same matrix as [K, N].
  // CHECK-LABEL: func.func @ds_matmul_unit_batched_weight
  // CHECK: matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_dram_sharded_program_config
  // CHECK-SAME: per_core_m = 1
  func.func @ds_matmul_unit_batched_weight(
      %act: tensor<1x1x32x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
      %weight: tensor<1x1x4096x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
      %other: tensor<1x1x32x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}) -> tensor<1x1x32x4096xbf16> {
    %0 = "ttir.matmul"(%act, %weight) : (tensor<1x1x32x4096xbf16>, tensor<1x1x4096x4096xbf16>) -> tensor<1x1x32x4096xbf16>
    %1 = "ttir.multiply"(%0, %other) : (tensor<1x1x32x4096xbf16>, tensor<1x1x32x4096xbf16>) -> tensor<1x1x32x4096xbf16>
    return %1 : tensor<1x1x32x4096xbf16>
  }

  // SwiGLU: the silu folds onto the multiply's operand, not into the matmul,
  // so the matmul can take the DS config. The fold is positional (operand A).
  // CHECK-LABEL: func.func @ds_matmul_swiglu
  // CHECK: "ttnn.matmul"
  // CHECK-SAME: matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_dram_sharded_program_config
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

  // Operands swapped: the silu folds onto operand B.
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
