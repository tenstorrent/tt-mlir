// RUN: ttmlir-opt %s | FileCheck %s

// Verify that the new `dtype` attribute on `ttnn.matmul` parses and prints
// roundtrip-stable.

#dram = #ttnn.buffer_type<dram>
#ttnn_layout_bf16 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout_bf16_b = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout_bfp8 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x8x!ttcore.tile<32x32, bfp_bf8>, #dram>, <interleaved>>

module attributes {} {
  // CHECK-LABEL: func.func @matmul_with_explicit_bfp8_dtype
  func.func @matmul_with_explicit_bfp8_dtype(
      %arg0: tensor<32x128xbf16, #ttnn_layout_bf16>,
      %arg1: tensor<128x256xbf16, #ttnn_layout_bf16_b>
  ) -> tensor<32x256xbf16, #ttnn_layout_bfp8> {
    // CHECK: "ttnn.matmul"(%arg0, %arg1)
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bfp_bf8>
    %0 = "ttnn.matmul"(%arg0, %arg1) <{
      transpose_a = false,
      transpose_b = false,
      dtype = #ttcore.supportedDataTypes<bfp_bf8>
    }> : (tensor<32x128xbf16, #ttnn_layout_bf16>, tensor<128x256xbf16, #ttnn_layout_bf16_b>) -> tensor<32x256xbf16, #ttnn_layout_bfp8>
    return %0 : tensor<32x256xbf16, #ttnn_layout_bfp8>
  }

  // CHECK-LABEL: func.func @matmul_without_dtype
  func.func @matmul_without_dtype(
      %arg0: tensor<32x128xbf16, #ttnn_layout_bf16>,
      %arg1: tensor<128x256xbf16, #ttnn_layout_bf16_b>
  ) -> tensor<32x256xbf16, #ttnn_layout_bf16> {
    // The dtype attribute is optional; ops without it must continue to parse.
    // CHECK: "ttnn.matmul"(%arg0, %arg1)
    // CHECK-NOT: dtype = #ttcore.supportedDataTypes
    %0 = "ttnn.matmul"(%arg0, %arg1) <{
      transpose_a = false,
      transpose_b = false
    }> : (tensor<32x128xbf16, #ttnn_layout_bf16>, tensor<128x256xbf16, #ttnn_layout_bf16_b>) -> tensor<32x256xbf16, #ttnn_layout_bf16>
    return %0 : tensor<32x256xbf16, #ttnn_layout_bf16>
  }
}
