// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=2 experimental-weight-dtype=bfp_bf8 enable-dram-sharded-matmul=true" -o %t %s
// RUN: FileCheck %s --input-file=%t

// An activation taller than one tile row must be declined at compile time.
//
// tt-metal's DRAM-sharded validation asserts TT_FATAL(M == 1) on the activation
// height in tiles, and a TT_FATAL is an uncatchable abort rather than a failure
// the op model can report. So this has to be rejected by the eligibility gate:
// deferring to tt-metal would turn a compile-time decline into a crash on
// silicon. This is the prefill / large-batch shape.
//
// Everything else here is identical to the eligible baseline, so the only reason
// DS is absent is M = 64 (2 tile rows).

module attributes {} {
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
