// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-optimizer=true enable-bfp8-conversion=true" --mlir-print-local-scope %s -o %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

module  {
  // CHECK-LABEL: func.func @conv2d_bf16_to_bfp8_const_eval_0
  // CHECK: "ttnn.prepare_conv2d_bias"
  // CHECK-SAME: tensor<1x1x1x64xbf16, #ttnn.ttnn_layout<{{.*}}, memref<1x64xbf16, #ttnn.buffer_type<system_memory
  // CHECK-SAME: -> tensor<1x1x1x64x!ttcore.tile<32x32, bfp_bf8

  // CHECK-LABEL: func.func @conv2d_bf16_to_bfp8_const_eval_1
  // CHECK: "ttnn.prepare_conv2d_weights"
  // CHECK-SAME: tensor<64x64x3x3xbf16, #ttnn.ttnn_layout<{{.*}}, memref<12288x3xbf16, #ttnn.buffer_type<system_memory
  // CHECK-SAME: -> tensor<1x1x576x64x!ttcore.tile<32x32, bfp_bf8

  // Weight needs to be in row major in function signature.
  // CHECK-LABEL: func.func @conv2d_bf16_to_bfp8
  // CHECK-SAME: tensor<64x64x3x3xbf16, #ttnn.ttnn_layout<{{.*}}, memref<12288x3xbf16, #ttnn.buffer_type<system_memory
  func.func @conv2d_bf16_to_bfp8(%input: tensor<16x32x32x64xbf16> {ttcore.argument_type = #ttcore.argument_type<input>}, %weight: tensor<64x64x3x3xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}, %bias: tensor<1x1x1x64xbf16> {ttcore.argument_type = #ttcore.argument_type<constant>}) -> tensor<16x30x30x64xbf16> {
    %0 = ttir.empty() : tensor<16x30x30x64xbf16>
    // CHECK: "ttnn.conv2d"
    // Conv2d config should have weights_dtype updated to bfp_bf8 since prepared weight and bias will be in bfp_bf8
    // CHECK-SAME: conv2d_config = #ttnn.conv2d_config<
    // CHECK-SAME: weights_dtype = bfp_bf8

    // Input should be in row major bf16
    // CHECK-SAME: tensor<1x1x16384x64x!ttcore.tile<32x32, bfp_bf8>, #ttnn.ttnn_layout<{{.*}}, memref<512x2x!ttcore.tile<32x32, bfp_bf8>, #ttnn.buffer_type<dram

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
