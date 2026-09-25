// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=2 experimental-weight-dtype=bfp_bf8 mock-system-desc-arch=blackhole enable-dram-sharded-matmul=true" -o %t %s
// RUN: FileCheck %s --input-file=%t

// The DS path declines a matmul whose result feeds a collective: no CCL has an
// op model, so the optimizer cannot cost the reshard it would put on the
// collective's critical path. llama_3_1_70b's TP=4 down projection; its
// in0_block_w fits at 7, so the geometry itself is eligible.

module attributes {} {
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
