// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=2 experimental-weight-dtype=bfp_bf8" -o %t %s
// RUN: FileCheck %s --input-file=%t

// A weight with a non-unit batch dim is a real batched matmul and is not
// DS-eligible.
//
// tt-metal serves that case with a different program config
// (MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig) which this path
// does not emit, so accepting a [2, 1, K, N] weight here would build a config
// for the wrong kernel.
//
// The activation deliberately stays at one tile row (M = 1*1*32 = 32) so this
// test isolates the weight-shape gate rather than tripping the M gate.

module attributes {} {
  // The positive anchor matters: with only a CHECK-NOT this test would still
  // pass if the matmul vanished or the pipeline broke upstream.
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
}
