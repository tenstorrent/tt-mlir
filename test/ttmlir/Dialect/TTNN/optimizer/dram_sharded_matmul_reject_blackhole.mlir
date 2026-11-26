// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=2 experimental-weight-dtype=bfp_bf8 mock-system-desc-arch=blackhole enable-dram-sharded-matmul=true" -o %t %s
// RUN: FileCheck %s --input-file=%t

// Shapes the DS gate declines only on the Blackhole mock (8 DRAM banks).

module attributes {} {
  // K = 11008 is 344 tiles, 43 per in0 core, a prime: the only block widths are
  // 43, whose in1 CB does not fit on 8 banks, and 1, which is below
  // kMinBlockWidth (qwen_2_5_3b's down projection). On 12 banks the shard is
  // narrower and width 43 fits.
  // CHECK-LABEL: func.func @ds_matmul_block_collapse
  // CHECK: "ttnn.matmul"
  // CHECK-NOT: dram_sharded_program_config
  func.func @ds_matmul_block_collapse(
      %act: tensor<32x11008xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
      %weight: tensor<11008x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
      %other: tensor<32x2048xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}) -> tensor<32x2048xbf16> {
    %0 = "ttir.matmul"(%act, %weight) : (tensor<32x11008xbf16>, tensor<11008x2048xbf16>) -> tensor<32x2048xbf16>
    %1 = "ttir.multiply"(%0, %other) : (tensor<32x2048xbf16>, tensor<32x2048xbf16>) -> tensor<32x2048xbf16>
    return %1 : tensor<32x2048xbf16>
  }

  // The result feeds a collective, which the optimizer cannot cost, so DS is
  // declined for the consumer; the geometry itself fits (in0_block_w = 7). The
  // eligible counterpart is dram_sharded_matmul_eligible_blackhole.mlir.
  // CHECK-LABEL: func.func @ds_matmul_feeds_all_reduce
  // CHECK: "ttnn.matmul"
  // CHECK-NOT: dram_sharded_program_config
  func.func @ds_matmul_feeds_all_reduce(
      %act: tensor<32x7168xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
      %weight: tensor<7168x8192xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}) -> tensor<32x8192xbf16> {
    %0 = "ttir.matmul"(%act, %weight) : (tensor<32x7168xbf16>, tensor<7168x8192xbf16>) -> tensor<32x8192xbf16>
    %1 = "ttir.all_reduce"(%0) <{cluster_axis = 0 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<32x8192xbf16>) -> tensor<32x8192xbf16>
    return %1 : tensor<32x8192xbf16>
  }
}
