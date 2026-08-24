// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=2 experimental-weight-dtype=bfp_bf8 mock-system-desc-arch=blackhole" -o %t %s
// RUN: FileCheck %s --input-file=%t

// The DS path declines when the CB budget collapses in0_block_w (see
// kMinBlockWidthFraction in MatmulProgramConfig.cpp).
//
// K = 11008 is 344 tiles, so K-per-core is 344/8 = 43 -- a prime, leaving 43 and
// 1 as the only legal block widths. On an 8-bank part the in1 CB at in0_block_w
// = 43 needs ~1.35 MB against a ~1.30 MB budget, so the search would otherwise
// settle on in0_block_w = 1: 344 serialized mcast+compute rounds instead of 8.
// That shape is qwen_2_5_3b's down-projection, measured at -28.1% e2e on p150.
//
// Blackhole specifically: with 12 banks the per-bank weight shard is narrower, the
// in1 CB fits at in0_block_w = 43, and the collapse never happens -- which is why
// none of the Wormhole benchmarks regressed.
//
// As with the disable-dram-sharded-matmul kill switch, the matmul must still get
// *some* program config; declining DS falls back to the 1D/2D mcast configs
// rather than disabling program-config selection entirely.

module attributes {} {
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
}
