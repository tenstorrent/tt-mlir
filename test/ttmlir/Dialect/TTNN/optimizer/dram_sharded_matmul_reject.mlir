// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=2 experimental-weight-dtype=bfp_bf8 enable-dram-sharded-matmul=true" -o %t %s
// RUN: FileCheck %s --input-file=%t

// Shapes the DS gate declines, each one step from the eligible baseline. The
// positive CHECK on the matmul keeps the CHECK-NOT from passing vacuously.

module attributes {} {
  // A non-unit weight batch dim needs the batched DS config, which this path
  // does not emit.
  // CHECK-LABEL: func.func @ds_matmul_batched_weight
  // CHECK: "ttnn.matmul"
  // CHECK-NOT: dram_sharded_program_config
  func.func @ds_matmul_batched_weight(
      %act: tensor<1x1x32x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
      %weight: tensor<2x1x4096x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
      %other: tensor<2x1x32x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}) -> tensor<2x1x32x4096xbf16> {
    %0 = "ttir.matmul"(%act, %weight) : (tensor<1x1x32x4096xbf16>, tensor<2x1x4096x4096xbf16>) -> tensor<2x1x32x4096xbf16>
    %1 = "ttir.multiply"(%0, %other) : (tensor<2x1x32x4096xbf16>, tensor<2x1x32x4096xbf16>) -> tensor<2x1x32x4096xbf16>
    return %1 : tensor<2x1x32x4096xbf16>
  }

  // A fused activation with no binary consumer stays on the matmul; a DS config
  // cannot carry it.
  // CHECK-LABEL: func.func @ds_matmul_fused_activation
  // CHECK: "ttnn.matmul"
  // CHECK-SAME: fused_activation = #ttnn.unary_with_param<op_type = silu>
  // CHECK-NOT: dram_sharded_program_config
  func.func @ds_matmul_fused_activation(
      %act: tensor<32x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
      %weight: tensor<4096x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}) -> tensor<32x4096xbf16> {
    %0 = "ttir.matmul"(%act, %weight) : (tensor<32x4096xbf16>, tensor<4096x4096xbf16>) -> tensor<32x4096xbf16>
    %1 = "ttir.silu"(%0) : (tensor<32x4096xbf16>) -> tensor<32x4096xbf16>
    return %1 : tensor<32x4096xbf16>
  }

  // K = 2880 is 90 tiles, not divisible by the 8 in0 cores (gpt-oss hidden size).
  // CHECK-LABEL: func.func @ds_matmul_k_not_divisible
  // CHECK: "ttnn.matmul"
  // CHECK-NOT: dram_sharded_program_config
  func.func @ds_matmul_k_not_divisible(
      %act: tensor<32x2880xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
      %weight: tensor<2880x2880xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
      %other: tensor<32x2880xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}) -> tensor<32x2880xbf16> {
    %0 = "ttir.matmul"(%act, %weight) : (tensor<32x2880xbf16>, tensor<2880x2880xbf16>) -> tensor<32x2880xbf16>
    %1 = "ttir.multiply"(%0, %other) : (tensor<32x2880xbf16>, tensor<32x2880xbf16>) -> tensor<32x2880xbf16>
    return %1 : tensor<32x2880xbf16>
  }

  // M = 64 is two tile rows; tt-metal asserts M == 1 uncatchably.
  // CHECK-LABEL: func.func @ds_matmul_m64
  // CHECK: "ttnn.matmul"
  // CHECK-NOT: dram_sharded_program_config
  func.func @ds_matmul_m64(
      %act: tensor<64x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
      %weight: tensor<4096x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
      %other: tensor<64x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}) -> tensor<64x4096xbf16> {
    %0 = "ttir.matmul"(%act, %weight) : (tensor<64x4096xbf16>, tensor<4096x4096xbf16>) -> tensor<64x4096xbf16>
    %1 = "ttir.multiply"(%0, %other) : (tensor<64x4096xbf16>, tensor<64x4096xbf16>) -> tensor<64x4096xbf16>
    return %1 : tensor<64x4096xbf16>
  }
}
