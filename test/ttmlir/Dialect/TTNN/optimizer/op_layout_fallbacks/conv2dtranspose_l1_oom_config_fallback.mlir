// REQUIRES: opmodel
// RUN: ttmlir-opt --ttcore-register-device --ttnn-operation-validation-and-fallback="max-fallback-attempts=35" -o %t %s
// RUN: FileCheck %s --input-file=%t
//
// This test verifies that when a ConvTranspose2d operation with conv2d_slice_config=<l1_full>
// causes an L1 OOM error, the optimizer fallback pass fixes it by replacing the slice config
// to use DRAM slicing instead.

#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>

module {
  func.func @test_conv2d_transpose_l1_oom(
    %arg0: tensor<1x1x65536x256xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 65536 + d1 * 65536 + d2, d3), <1x1>, memref<2048x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>,
    %arg1: tensor<256x256x3x3xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 768 + d1 * 3 + d2, d3), <1x1>, memref<196608x3xbf16, #system_memory>>>,
    %arg2: tensor<1x1x1x256xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 + d1 + d2, d3), <1x1>, memref<1x256xbf16, #system_memory>>>
  ) -> tensor<1x1x261121x256xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 261152 + d1 * 261152 + d2, d3), <1x1>, memref<8161x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>> {
    %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device

    // ConvTranspose2d with conv2d_slice_config=<l1_full> which causes L1 OOM.
    // The fallback should replace the slice config with DRAM slicing.
    // CHECK: ttnn.conv_transpose2d
    // CHECK: conv2d_slice_config = #ttnn.conv2d_slice_config<dram
    %result = "ttnn.conv_transpose2d"(%arg0, %arg1, %arg2, %0) <{
      batch_size = 1 : i32,
      conv2d_config = #ttnn.conv2d_config<
        weights_dtype = bf16,
        activation = <op_type = relu>,
        deallocate_activation = true,
        act_block_h_override = 0,
        enable_kernel_stride_folding = false,
        config_tensors_in_dram = true
      >,
      conv2d_slice_config = #ttnn.conv2d_slice_config<l1_full, 0>,
      dilation = array<i32: 1, 1>,
      dtype = #ttcore.supportedDataTypes<bf16>,
      groups = 1 : i32,
      in_channels = 256 : i32,
      input_height = 256 : i32,
      input_width = 256 : i32,
      kernel_size = array<i32: 3, 3>,
      out_channels = 256 : i32,
      output_padding = array<i32: 0, 0>,
      padding = array<i32: 1, 1>,
      stride = array<i32: 2, 2>
    }> : (
      tensor<1x1x65536x256xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 65536 + d1 * 65536 + d2, d3), <1x1>, memref<2048x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>,
      tensor<256x256x3x3xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 768 + d1 * 3 + d2, d3), <1x1>, memref<196608x3xbf16, #system_memory>>>,
      tensor<1x1x1x256xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 + d1 + d2, d3), <1x1>, memref<1x256xbf16, #system_memory>>>,
      !ttnn.device
    ) -> tensor<1x1x261121x256xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 261152 + d1 * 261152 + d2, d3), <1x1>, memref<8161x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>

    return %result : tensor<1x1x261121x256xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 261152 + d1 * 261152 + d2, d3), <1x1>, memref<8161x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>
  }
}
