// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=2 experimental-weight-dtype=bfp_bf8 enable-dram-sharded-matmul=true" -o %t %s
// RUN: FileCheck %s --input-file=%t

// Baseline DRAM-sharded (DS) matmul: a decode-shaped projection with a bfp8
// constant weight. The activation is exactly one tile row (batch 32), K and N
// are tile-aligned, and K in tiles (128) is divisible by the in0 core count (8),
// so every eligibility gate passes and the optimizer offers the DS config.
//
// per_core_m must be 1: tt-metal asserts M == per_core_M and M == 1 for the
// DRAM-sharded program config.

module attributes {} {
  // CHECK-LABEL: func.func @ds_matmul_m32
  // CHECK: matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_dram_sharded_program_config
  // CHECK-SAME: per_core_m = 1
  func.func @ds_matmul_m32(
      %act: tensor<32x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
      %weight: tensor<4096x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
      %other: tensor<32x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}) -> tensor<32x4096xbf16> {
    %0 = "ttir.matmul"(%act, %weight) : (tensor<32x4096xbf16>, tensor<4096x4096xbf16>) -> tensor<32x4096xbf16>
    // Consumer op to form an L1 chain, matching optimizer/matmul_program_config.mlir.
    %1 = "ttir.multiply"(%0, %other) : (tensor<32x4096xbf16>, tensor<32x4096xbf16>) -> tensor<32x4096xbf16>
    return %1 : tensor<32x4096xbf16>
  }
}
