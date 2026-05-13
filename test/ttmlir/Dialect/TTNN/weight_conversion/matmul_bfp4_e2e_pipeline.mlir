// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-optimizer=true enable-const-eval=true experimental-weight-dtype=bfp_bf4" %s | FileCheck %s
// REQUIRES: opmodel

// End-to-end test: TTIR matmul through the full backend pipeline with BFP4
// weight conversion, optimizer, and const-eval enabled.
// Verifies that the host-side typecast chain (from_device -> typecast -> to_device)
// survives all pipeline passes and the typecast operates on a host tensor.

module {
  func.func @matmul_bfp4_e2e(
    %arg0: tensor<64x128xbf16>,
    %arg1: tensor<128x64xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>}
  ) -> tensor<64x64xbf16> {
    %0 = "ttir.matmul"(%arg0, %arg1) : (tensor<64x128xbf16>, tensor<128x64xbf16>) -> tensor<64x64xbf16>
    return %0 : tensor<64x64xbf16>
  }
}

// The weight conversion should be hoisted into a const_eval function.
// CHECK: func.func private @matmul_bfp4_e2e_const_eval_0
// CHECK-SAME: tt.function_type = "const_eval"

// Inside const_eval: from_device brings tensor to host, typecast on host, to_device sends back.
// CHECK: "ttnn.from_device"
// CHECK: "ttnn.typecast"
// CHECK-SAME: dtype = #ttcore.supportedDataTypes<bfp_bf4>
// CHECK: "ttnn.to_device"

// The forward function uses load_cached and matmul.
// CHECK: func.func @matmul_bfp4_e2e
// CHECK: ttcore.load_cached
// CHECK: "ttnn.matmul"
