// REQUIRES: opmodel
// RUN: ttmlir-opt --ttcore-register-device --ttcore-mark-functions-as-forward --ttnn-operation-validation-and-fallback %s -o %t.mlir
// RUN: FileCheck %s --input-file %t.mlir

// Test that the post-optimizer validation analysis automatically detects
// and handles conv_transpose2d operations that fail validation.

#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout_input = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 16384 + d1 * 16384 + d2, d3), <1x1>, memref<8x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout_weight = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 2 + d2, d3), <1x1>, memref<16384x2xbf16, #system_memory>>
#ttnn_layout_bias = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 + d1 + d2, d3), <1x1>, memref<1x64xbf16, #system_memory>>
#ttnn_layout_output = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 65536 + d1 * 65536 + d2, d3), <1x1>, memref<2048x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module attributes {} {
  func.func @conv_transpose2d_validation(%arg0: tensor<1x1x16384x128xbf16, #ttnn_layout_input>,
                                         %arg1: tensor<128x64x2x2xbf16, #ttnn_layout_weight>,
                                         %arg2: tensor<1x1x1x64xbf16, #ttnn_layout_bias>) -> tensor<1x1x65536x64xbf16, #ttnn_layout_output> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device

    // The op validation pass should detect that conv_transpose2d may fail validation
    // and insert appropriate fallback transformations if needed.

    // CHECK: %[[RES:.*]] = "ttnn.conv_transpose2d"
    // CHECK-SAME: batch_size = 1 : i32
    // CHECK-SAME: conv2d_config = #ttnn.conv2d_config<weights_dtype = bf16, deallocate_activation = true, act_block_h_override = {{[0-9]+}}, enable_kernel_stride_folding = false>
    // CHECK-SAME: dilation = array<i32: 1, 1>
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK-SAME: groups = 1 : i32
    // CHECK-SAME: in_channels = 128 : i32
    // CHECK-SAME: input_height = 128 : i32
    // CHECK-SAME: input_width = 128 : i32
    // CHECK-SAME: kernel_size = array<i32: 2, 2>
    // CHECK-SAME: out_channels = 64 : i32
    // CHECK-SAME: output_padding = array<i32: 0, 0>
    // CHECK-SAME: padding = array<i32: 0, 0>
    // CHECK-SAME: stride = array<i32: 2, 2>

    %1 = "ttnn.conv_transpose2d"(%arg0, %arg1, %arg2, %0) <{
      batch_size = 1 : i32,
      conv2d_config = #ttnn.conv2d_config<weights_dtype = bf16, deallocate_activation = true, act_block_h_override = 0, enable_kernel_stride_folding = false>,
      dilation = array<i32: 1, 1>,
      dtype = #ttcore.supportedDataTypes<bf16>,
      groups = 1 : i32,
      in_channels = 128 : i32,
      input_height = 128 : i32,
      input_width = 128 : i32,
      kernel_size = array<i32: 2, 2>,
      out_channels = 64 : i32,
      output_padding = array<i32: 0, 0>,
      padding = array<i32: 0, 0>,
      stride = array<i32: 2, 2>
    }> : (tensor<1x1x16384x128xbf16, #ttnn_layout_input>,
         tensor<128x64x2x2xbf16, #ttnn_layout_weight>,
         tensor<1x1x1x64xbf16, #ttnn_layout_bias>,
         !ttnn.device) -> tensor<1x1x65536x64xbf16, #ttnn_layout_output>

    return %1 : tensor<1x1x65536x64xbf16, #ttnn_layout_output>
  }
}
