// RUN: not ttmlir-opt --ttnn-deallocate --split-input-file %s 2>&1 | FileCheck %s
// Test that verifies the sanity check for conv2d operations with deallocate_activation=true
// The conv2d must be the last user of its input tensor to prevent use-after-free

#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0, d1, d2, d3), <1x1>, memref<32x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0, d1, d2, d3), <1x1>, memref<18x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0, d1, d2, d3), <1x1>, memref<1x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module {
  func.func @forward(%input: tensor<1x1x1024x64xbf16, #ttnn_layout>, %weight: tensor<1x1x576x64xbf16, #ttnn_layout1>, %bias: tensor<1x1x1x64xbf16, #ttnn_layout2>) -> tensor<1x1x1024x64xbf16, #ttnn_layout> {
    %device = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device

    // This conv2d has deallocate_activation=true, which means it will deallocate %input
    // However, it's not the last user because the add operation also uses %input
    // CHECK: error: 'ttnn.conv2d' op Conv2dOp with `deallocate_activation` set to true must be the last user of the input tensor.
    %conv_out = "ttnn.conv2d"(%input, %weight, %bias, %device) <{
      in_channels = 64: i32,
      out_channels = 64: i32,
      batch_size = 1: i32,
      input_height = 32: i32,
      input_width = 32: i32,
      kernel_size = array<i32: 3, 3>,
      stride = array<i32: 1, 1>,
      padding = array<i32: 1, 1>,
      dilation = array<i32: 1, 1>,
      groups = 1: i32,
      dtype = #ttcore.supportedDataTypes<bf16>,
      conv2d_config = #ttnn.conv2d_config<weights_dtype = bf16, deallocate_activation = true, act_block_h_override = 0>,
      conv2d_slice_config = #ttnn.conv2d_slice_config<l1_full, 0>
    }> : (tensor<1x1x1024x64xbf16, #ttnn_layout>, tensor<1x1x576x64xbf16, #ttnn_layout1>, tensor<1x1x1x64xbf16, #ttnn_layout2>, !ttnn.device) -> tensor<1x1x1024x64xbf16, #ttnn_layout>

    // This operation also uses %input, creating a use-after-free since conv2d above deallocates it
    %result = "ttnn.add"(%input, %conv_out) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<1x1x1024x64xbf16, #ttnn_layout>, tensor<1x1x1024x64xbf16, #ttnn_layout>) -> tensor<1x1x1024x64xbf16, #ttnn_layout>

    return %result : tensor<1x1x1024x64xbf16, #ttnn_layout>
  }
}
