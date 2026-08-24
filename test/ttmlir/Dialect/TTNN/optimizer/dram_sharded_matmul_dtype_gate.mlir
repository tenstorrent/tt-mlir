// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=2 experimental-weight-dtype=bfp_bf8" -o %t %s
// RUN: FileCheck %s --input-file=%t --check-prefix=BFP8
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=2" -o %t2 %s
// RUN: FileCheck %s --input-file=%t2 --check-prefix=BF16

// The DS path is offered for bfp4/bfp8 weights only. Identical IR, two weight
// dtypes.
//
// This is a bandwidth policy, not a legality constraint: tt-metal imposes no
// dtype TT_FATAL on the DRAM-sharded config, and bf16 weights do run. But DS
// streams the weights out of DRAM, so bf16 moves 2x the bytes of bfp8 and 4x
// bfp4 -- the regime where 1D-mcast wins -- and the optimizer has no runtime
// estimate with which to rank the two. The conservative set stays hardcoded.

module attributes {} {
  // BFP8-LABEL: func.func @ds_matmul_dtype
  // BFP8: matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_dram_sharded_program_config

  // BF16-LABEL: func.func @ds_matmul_dtype
  // BF16-NOT: dram_sharded_program_config
  func.func @ds_matmul_dtype(
      %act: tensor<32x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
      %weight: tensor<4096x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
      %other: tensor<32x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}) -> tensor<32x4096xbf16> {
    %0 = "ttir.matmul"(%act, %weight) : (tensor<32x4096xbf16>, tensor<4096x4096xbf16>) -> tensor<32x4096xbf16>
    %1 = "ttir.multiply"(%0, %other) : (tensor<32x4096xbf16>, tensor<32x4096xbf16>) -> tensor<32x4096xbf16>
    return %1 : tensor<32x4096xbf16>
  }
}
