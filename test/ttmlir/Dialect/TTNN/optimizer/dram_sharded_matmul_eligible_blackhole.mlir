// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=2 experimental-weight-dtype=bfp_bf8 mock-system-desc-arch=blackhole enable-dram-sharded-matmul=true" -o %t %s
// RUN: FileCheck %s --input-file=%t
//
// tt-metal#57022: the stateful op-model query trips on the Metal 2.0 DS
// factory's borrowed-memory DFBs. Remove once the fix is uplifted.
// XFAIL: *

// Control for dram_sharded_matmul_reject_blackhole.mlir: the same shape as the
// collective case with an ordinary consumer takes the DS path.

module attributes {} {
  // CHECK-LABEL: func.func @ds_matmul_no_ccl
  // CHECK: "ttnn.matmul"
  // CHECK-SAME: dram_sharded_program_config
  func.func @ds_matmul_no_ccl(
      %act: tensor<32x7168xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
      %weight: tensor<7168x8192xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
      %other: tensor<32x8192xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}) -> tensor<32x8192xbf16> {
    %0 = "ttir.matmul"(%act, %weight) : (tensor<32x7168xbf16>, tensor<7168x8192xbf16>) -> tensor<32x8192xbf16>
    %1 = "ttir.multiply"(%0, %other) : (tensor<32x8192xbf16>, tensor<32x8192xbf16>) -> tensor<32x8192xbf16>
    return %1 : tensor<32x8192xbf16>
  }
}
