// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=2 experimental-weight-dtype=bfp_bf8 enable-dram-sharded-matmul=false" -o %t %s
// RUN: FileCheck %s --input-file=%t --implicit-check-not='"ttnn.silu"'

// enable-dram-sharded-matmul=false (the default): no DS config, and the
// activation fusing behaves as it did before DS.

module attributes {} {
  // The eligible baseline with the option off.
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

  // The DS-off half of the SwiGLU case: the silu folds onto the matmul as
  // fused_activation and the multiply keeps empty activation lists.
  // CHECK-LABEL: func.func @ds_matmul_swiglu_ds_off
  // CHECK: "ttnn.matmul"
  // CHECK-SAME: fused_activation = #ttnn.unary_with_param<op_type = silu>
  // CHECK-NOT: dram_sharded_program_config
  // CHECK: "ttnn.multiply"
  // CHECK-SAME: input_tensor_a_activations = []
  // CHECK-SAME: input_tensor_b_activations = []
  func.func @ds_matmul_swiglu_ds_off(
      %act: tensor<32x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
      %gate_w: tensor<4096x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>},
      %up: tensor<32x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}) -> tensor<32x4096xbf16> {
    %0 = "ttir.matmul"(%act, %gate_w) : (tensor<32x4096xbf16>, tensor<4096x4096xbf16>) -> tensor<32x4096xbf16>
    %1 = "ttir.silu"(%0) : (tensor<32x4096xbf16>) -> tensor<32x4096xbf16>
    %2 = "ttir.multiply"(%1, %up) : (tensor<32x4096xbf16>, tensor<32x4096xbf16>) -> tensor<32x4096xbf16>
    return %2 : tensor<32x4096xbf16>
  }
}
