// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=2 experimental-weight-dtype=bfp_bf8" -o %t %s
// RUN: FileCheck %s --input-file=%t

// A matmul that still carries a fused activation must be declined by the DS gate.
//
// Here the silu has no binary consumer to fold into, so it stays on the matmul
// and survives to the optimizer. A DS config's fused_activation is always null,
// so the op model would validate the config without the activation while the
// runtime hands op->activation() to ::ttnn::matmul alongside it.
//
// Declining hands the op to a 1D/2D mcast config, which folds the activation
// into its own fused_activation. Everything else matches the eligible baseline,
// so the fused activation is the only reason DS is absent.

module attributes {} {
  // CHECK-LABEL: func.func @ds_matmul_fused_activation
  // CHECK: "ttnn.matmul"
  // CHECK-SAME: fused_activation = #ttnn.unary_with_param<op_type = silu>
  // CHECK-NOT: dram_sharded_program_config
  func.func @ds_matmul_fused_activation(
      %act: tensor<32x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<input>},
      %weight: tensor<4096x4096xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}) -> tensor<32x4096xbf16> {
    %0 = "ttir.matmul"(%act, %weight) : (tensor<32x4096xbf16>, tensor<4096x4096xbf16>) -> tensor<32x4096xbf16>
    %1 = "ttir.silu"(%0) : (tensor<32x4096xbf16>) -> tensor<32x4096xbf16>
    return %1 : tensor<32x4096xbf16>
  }
}
