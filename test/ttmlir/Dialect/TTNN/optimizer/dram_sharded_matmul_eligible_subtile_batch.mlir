// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=2 experimental-weight-dtype=bfp_bf8 enable-dram-sharded-matmul=true" -o %t %s
// RUN: FileCheck %s --input-file=%t

// A sub-tile decode batch is still one tile row, so it is DS-eligible.
//
// tt-metal's constraint is on the activation height in *tiles*
// (TT_FATAL(M == 1)), and it pads a 1..31-row activation up to one tile row, so
// the gate is divUp(M, 32) == 1 rather than a tile-alignment check on M. Any
// batch in 1..32 qualifies, and per_core_M comes out as 1 for all of them
// (computeShardParams rounds up, so a sub-tile M cannot truncate to a
// degenerate 0).

module attributes {} {
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
}
