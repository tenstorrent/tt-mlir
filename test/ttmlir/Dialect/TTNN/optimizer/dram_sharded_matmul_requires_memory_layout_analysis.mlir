// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=1 experimental-weight-dtype=bfp_bf8 enable-dram-sharded-matmul=true" -o %t.o1 %s
// RUN: FileCheck %s --input-file=%t.o1 --check-prefix=O1
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=2 experimental-weight-dtype=bfp_bf8 enable-dram-sharded-matmul=true" -o %t.o2 %s
// RUN: FileCheck %s --input-file=%t.o2 --check-prefix=O2

module attributes {} {
  // O1-LABEL: func.func @ds_matmul_requires_mla
  // O1: "ttnn.matmul"
  // O1-SAME: activation = "silu"
  // O1-NOT: dram_sharded_program_config
  // O1: "ttnn.multiply"
  // O1-SAME: input_tensor_a_activations = []
  // O1-SAME: input_tensor_b_activations = []

  // O2-LABEL: func.func @ds_matmul_requires_mla
  // O2: "ttnn.matmul"
  // O2-SAME: matmul_program_config = #ttnn.matmul_multi_core_reuse_multi_cast_dram_sharded_program_config
  // O2: "ttnn.multiply"
  // O2-SAME: input_tensor_a_activations = [#ttnn.unary_with_param<op_type = silu>]
  func.func @ds_matmul_requires_mla(
      %act: tensor<32x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
      %gate_w: tensor<4096x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
      %up: tensor<32x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}) -> tensor<32x4096xbf16> {
    %0 = "ttir.matmul"(%act, %gate_w) : (tensor<32x4096xbf16>, tensor<4096x4096xbf16>) -> tensor<32x4096xbf16>
    %1 = "ttir.silu"(%0) : (tensor<32x4096xbf16>) -> tensor<32x4096xbf16>
    %2 = "ttir.multiply"(%1, %up) : (tensor<32x4096xbf16>, tensor<32x4096xbf16>) -> tensor<32x4096xbf16>
    return %2 : tensor<32x4096xbf16>
  }
}
