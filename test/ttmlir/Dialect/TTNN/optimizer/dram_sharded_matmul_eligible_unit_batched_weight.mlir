// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=2 experimental-weight-dtype=bfp_bf8 enable-dram-sharded-matmul=true" -o %t %s
// RUN: FileCheck %s --input-file=%t

// A [1, 1, K, N] weight is the same matrix as [K, N] and is DS-eligible.
//
// ttnn models routinely hold projection weights with leading unit batch dims,
// and such a weight has the same element count, the same tile grid and the same
// collapsed 2-D memref in the layout as the rank-2 form. Only a *non-unit*
// leading dim is a real batched matmul (see
// dram_sharded_matmul_reject_batched_weight.mlir).

module attributes {} {
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
}
