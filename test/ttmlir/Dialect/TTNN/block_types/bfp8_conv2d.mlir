// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-bfp8-conversion=true" --mlir-print-local-scope %s | FileCheck %s

module  {
  // CHECK-LABEL: func.func @conv2d_bf16_to_bfp8

  // Weight needs to be in row major in function signature.
  // CHECK-SAME: tensor<64x64x3x3xbf16, #ttnn.ttnn_layout<{{.*}}, memref<12288x3xbf16, #ttnn.buffer_type<system_memory
  func.func @conv2d_bf16_to_bfp8(%input: tensor<16x32x32x64xbf16>, %weight: tensor<64x64x3x3xbf16>, %bias: tensor<1x1x1x64xbf16>) -> tensor<16x30x30x64xbf16> {
    %0 = ttir.empty() : tensor<16x30x30x64xbf16>
    // CHECK: ttnn.conv2d

    // Weight should be on host row major bf16
    // CHECK-SAME: tensor<64x64x3x3xbf16, #ttnn.ttnn_layout{{.*}}, memref<12288x3xbf16, #ttnn.buffer_type<system_memory

    // Bias should be on host row major bf16
    // CHECK-SAME: tensor<1x1x1x64xbf16, #ttnn.ttnn_layout{{.*}}, memref<1x64xbf16, #ttnn.buffer_type<system_memory

    // Output should be in tile and in bfp_bf8
    // CHECK-SAME: -> tensor<1x1x14400x64x!ttcore.tile<32x32, bfp_bf8
    %1 = "ttir.conv2d"(%input, %weight, %bias, %0)
            <{
              stride = 1: i32,
              padding = 0: i32,
              dilation = 1: i32,
              groups = 1: i32
            }> : (tensor<16x32x32x64xbf16>, tensor<64x64x3x3xbf16>, tensor<1x1x1x64xbf16>, tensor<16x30x30x64xbf16>) -> tensor<16x30x30x64xbf16>
    return %1 : tensor<16x30x30x64xbf16>
  }
}
