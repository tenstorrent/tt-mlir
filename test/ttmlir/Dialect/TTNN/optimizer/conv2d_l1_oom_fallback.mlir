// REQUIRES: opmodel
// RUN: ttmlir-opt --ttcore-register-device --ttcore-mark-functions-as-forward --ttnn-operation-validation-and-fallback -o %t %s
// RUN: FileCheck %s --input-file=%t
//
// This test verifies that when a Conv2d operation with conv2d_slice_config=<l1_full>
// causes L1 OOM, the optimizer fallback pass fixes by replacing the slice config
// to allow DRAM usage instead.

#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>

module {
  func.func @test_conv2d_l1_oom(
    %arg0: tensor<1x1x852800x3xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 852800 + d1 * 852800 + d2, d3), <1x1>, memref<26650x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>,
    %arg1: tensor<64x3x7x7xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 21 + d1 * 7 + d2, d3), <1x1>, memref<1344x7xbf16, #system_memory>>>,
    %arg2: tensor<1x1x1x64xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 + d1 + d2, d3), <1x1>, memref<1x64xbf16, #system_memory>>>
  ) -> tensor<1x1x213200x64xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 213216 + d1 * 213216 + d2, d3), <1x1>, memref<6663x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>> {
    %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device

    // Conv2d with conv2d_slice_config=<l1_full> which will cause L1 OOM
    // The fallback should replace this attribute
    // CHECK: ttnn.conv2d
    // CHECK-NOT: conv2d_slice_config = #ttnn.conv2d_slice_config<l1_full
    %result = "ttnn.conv2d"(%arg0, %arg1, %arg2, %0) <{
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
      input_height = 800 : i32,
      input_width = 1066 : i32,
      kernel_size = array<i32: 7, 7>,
      out_channels = 64 : i32,
      padding = array<i32: 3, 3, 3, 3>,
      stride = array<i32: 2, 2>
    }> : (
      tensor<1x1x852800x3xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 852800 + d1 * 852800 + d2, d3), <1x1>, memref<26650x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>,
      tensor<64x3x7x7xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 21 + d1 * 7 + d2, d3), <1x1>, memref<1344x7xbf16, #system_memory>>>,
      tensor<1x1x1x64xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 + d1 + d2, d3), <1x1>, memref<1x64xbf16, #system_memory>>>,
      !ttnn.device
    ) -> tensor<1x1x213200x64xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 213216 + d1 * 213216 + d2, d3), <1x1>, memref<6663x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>

    return %result : tensor<1x1x213200x64xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 213216 + d1 * 213216 + d2, d3), <1x1>, memref<6663x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>
  }
}
