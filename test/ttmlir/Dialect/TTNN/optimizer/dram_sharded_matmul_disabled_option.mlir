// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=2 experimental-weight-dtype=bfp_bf8 disable-dram-sharded-matmul=true" -o %t %s
// RUN: FileCheck %s --input-file=%t

// The disable-dram-sharded-matmul kill switch suppresses the DS path.
//
// This is the same IR as dram_sharded_matmul_eligible_m32.mlir, which does get a
// DS config, so the only difference is the option. buildDSPlan is the single
// choke point for the DS path, which is what makes one check here sufficient:
// both entry points need a plan -- buildDRAMShardingHint for the output hint
// (the only route by which getOutputHints reaches DS) and
// getExtraInputReshardCandidates for the input candidates -- and the transform
// and hint-validation paths only ever see a config one of those produced.
//
// The matmul must still get *some* program config -- the option falls back to the
// 1D/2D mcast configs rather than disabling program-config selection entirely.

module attributes {} {
  // CHECK-LABEL: func.func @ds_matmul_disabled
  // CHECK: "ttnn.matmul"
  // CHECK-NOT: dram_sharded_program_config
  func.func @ds_matmul_disabled(
      %act: tensor<32x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
      %weight: tensor<4096x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
      %other: tensor<32x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}) -> tensor<32x4096xbf16> {
    %0 = "ttir.matmul"(%act, %weight) : (tensor<32x4096xbf16>, tensor<4096x4096xbf16>) -> tensor<32x4096xbf16>
    %1 = "ttir.multiply"(%0, %other) : (tensor<32x4096xbf16>, tensor<32x4096xbf16>) -> tensor<32x4096xbf16>
    return %1 : tensor<32x4096xbf16>
  }
}
