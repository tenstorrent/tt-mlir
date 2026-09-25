// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=2 experimental-weight-dtype=bfp_bf8 enable-dram-sharded-matmul=true" -o %t %s
// RUN: FileCheck %s --input-file=%t

// Known limitation, pinned deliberately: the DS activation is width-sharded
// across a fixed 8 in0 cores, so K in tiles must be divisible by 8.
//
// K = 2880 is 90 tiles, and 90 % 8 != 0, so this shape gets no DS config even
// though it is otherwise a textbook decode projection (this is the gpt-oss hidden
// size). Deriving the in0 core count from K would admit it -- 90 is divisible by
// 9, 6, 5 and more -- but picking *which* divisor is a cost question the
// optimizer cannot answer yet: it has no runtime estimate, and the candidates are
// all legal. Rather than bake in an arbitrary pick, the gate stays at 8.
//
// The gate also keeps computeShardParams' kTiles % numIn0Cores assertion true by
// construction instead of by caller contract, which matters because that assert
// is a no-op in a release build.
//
// If this test starts failing because DS *is* now offered, that is the in0-core
// search landing -- update it to assert the chosen split instead of its absence.

module attributes {} {
  // The positive anchor matters: with only a CHECK-NOT this test would still
  // pass if the matmul vanished or the pipeline broke upstream.
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
}
