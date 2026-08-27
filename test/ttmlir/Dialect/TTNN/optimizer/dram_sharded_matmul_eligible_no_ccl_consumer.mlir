// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=2 experimental-weight-dtype=bfp_bf8 mock-system-desc-arch=blackhole" -o %t %s
// RUN: FileCheck %s --input-file=%t

// Control for dram_sharded_matmul_reject_ccl_consumer: the identical shape with
// an ordinary consumer still takes the DS path, so the decline there is
// attributable to the collective and not to the geometry.

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
