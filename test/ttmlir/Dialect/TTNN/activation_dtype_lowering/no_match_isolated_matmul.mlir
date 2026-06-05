// RUN: ttmlir-opt --ttnn-ccl-activation-dtype-lowering %s | FileCheck %s

// A matmul whose result does not flow into any CCL op must be left untouched.
// The strict matchers should under-trigger by design.

#dram = #ttnn.buffer_type<dram>
#bf16_a = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#bf16_b = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#bf16_out = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module attributes {} {
  // CHECK-LABEL: func.func @matmul_no_ccl_downstream
  func.func @matmul_no_ccl_downstream(
      %arg0: tensor<32x128xbf16, #bf16_a>,
      %arg1: tensor<128x256xbf16, #bf16_b>
  ) -> tensor<32x256xbf16, #bf16_out> {
    // CHECK: "ttnn.matmul"
    // CHECK-NOT: bfp_bf8
    // CHECK-SAME: -> tensor<32x256xbf16
    %0 = "ttnn.matmul"(%arg0, %arg1) <{transpose_a = false, transpose_b = false}>
        : (tensor<32x128xbf16, #bf16_a>, tensor<128x256xbf16, #bf16_b>)
       -> tensor<32x256xbf16, #bf16_out>
    return %0 : tensor<32x256xbf16, #bf16_out>
  }
}
