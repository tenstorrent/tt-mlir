// REQUIRES: opmodel
// RUN: ttmlir-opt --ttcore-register-device --ttnn-operation-validation-and-fallback="max-fallback-attempts=35" -o %t %s
// RUN: FileCheck %s --input-file=%t
//
// This test verifies that when a ConvTranspose2d operation with conv2d_slice_config=<l1_full>
// doesn't cause L1 OOM, but has an error that requires the optimizer fallback pass fix by replacing the slice config
// to allow DRAM usage instead.

#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>

module {
  func.func @test_conv2d_transpose_l1_oom(
    %arg0: tensor<1x1x17589x3xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 17600 + d1 * 17600 + d2, d3), <1x1>, memref<550x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>,
    %arg1: tensor<3x640x63x63xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 189 + d1 * 63 + d2, d3), <1x1>, memref<120960x63xbf16, #system_memory>>>,
    %arg2: tensor<1x1x1x640xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 + d1 + d2, d3), <1x1>, memref<1x640xbf16, #system_memory>>>
  ) -> tensor<1x1x70356x640xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 70368 + d1 * 70368 + d2, d3), <1x1>, memref<2199x20x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>> {
    %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device

    // Conv2d with conv2d_slice_config=<l1_full> which will cause L1 OOM
    // The fallback should replace this attribute
    // CHECK: ttnn.conv_transpose2d
    // CHECK: conv2d_slice_config = #ttnn.conv2d_slice_config<l1
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
      in_channels = 3 : i32,
      input_height = 533 : i32,
      input_width = 33 : i32,
      kernel_size = array<i32: 63, 63>,
      out_channels = 640 : i32,
      output_padding = array<i32: 0, 0>,
      padding = array<i32: 31, 31>,
      stride = array<i32: 2, 2>
    }> : (
      tensor<1x1x17589x3xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 17600 + d1 * 17600 + d2, d3), <1x1>, memref<550x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>,
      tensor<3x640x63x63xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 189 + d1 * 63 + d2, d3), <1x1>, memref<120960x63xbf16, #system_memory>>>,
      tensor<1x1x1x640xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 + d1 + d2, d3), <1x1>, memref<1x640xbf16, #system_memory>>>,
      !ttnn.device
    ) -> tensor<1x1x70356x640xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 70368 + d1 * 70368 + d2, d3), <1x1>, memref<2199x20x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>

    return %result : tensor<1x1x70356x640xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 70368 + d1 * 70368 + d2, d3), <1x1>, memref<2199x20x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>
  }
}
