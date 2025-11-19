// RUN: ttmlir-opt --ttnn-weight-bfp8-conversion="experimental-bfp8-weights=true" %s | FileCheck %s

// Test that the BFP8 weight conversion pass correctly:
// 1. Sets output_dtype attribute of prepare_conv2d_weights to bfp_bf8
// 2. Updates the result type of prepare_conv2d_weights to have bfp_bf8 element type
// 3. Sets weights_dtype in conv2d_config to bfp_bf8
// 4. Keeps the output of conv2d as bf16 (unchanged)
// 5. Keeps the activation input dtype of conv2d as bf16 (unchanged)

#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>

module attributes {} {
  func.func @test_conv2d_bfp8_weights(%arg0: tensor<64x64x3x3xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 192 + d1 * 3 + d2, d3), <1x1>, memref<12288x3xbf16, #system_memory>>> {ttcore.argument_type = #ttcore.argument_type<parameter>}, %arg1: tensor<1x1x50176x64xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 50176 + d1 * 50176 + d2, d3), <1x1>, memref<1568x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>, %arg2: !ttnn.device) -> tensor<1x1x48400x64xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 48416 + d1 * 48416 + d2, d3), <1x1>, memref<1513x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>> {

    // CHECK-LABEL: func.func @test_conv2d_bfp8_weights

    // CHECK: "ttnn.prepare_conv2d_weights"
    // CHECK-SAME: output_dtype = #ttcore.supportedDataTypes<bfp_bf8>
    // CHECK-SAME: -> tensor<1x1x576x64x!ttcore.tile<32x32, bfp_bf8>,
    %0 = "ttnn.prepare_conv2d_weights"(%arg0, %arg2) <{
      batch_size = 1 : i32,
      conv2d_config = #ttnn.conv2d_config<
        weights_dtype = bf16,
        deallocate_activation = false,
        reallocate_halo_output = false,
        act_block_h_override = 0,
        act_block_w_div = 1,
        reshard_if_not_optimal = false,
        override_sharding_config = false,
        transpose_shards = false,
        output_layout = tile,
        enable_act_double_buffer = false,
        enable_weights_double_buffer = false,
        in_place = false,
        enable_kernel_stride_folding = false>,
      conv2d_slice_config = #ttnn.conv2d_slice_config<l1_full, 0>,
      dilation = array<i32: 2, 2>,
      groups = 1 : i32,
      has_bias = false,
      in_channels = 64 : i32,
      input_dtype = #ttcore.supportedDataTypes<bf16>,
      input_height = 224 : i32,
      input_memory_config = #ttnn.memory_config<#dram, <interleaved>>,
      input_tensor_layout = #ttnn.layout<tile>,
      input_width = 224 : i32,
      kernel_size = array<i32: 3, 3>,
      out_channels = 64 : i32,
      output_dtype = #ttcore.supportedDataTypes<bf16>,
      padding = array<i32: 0, 0, 0, 0>,
      stride = array<i32: 1, 1>,
      weights_format = "OIHW"
    }> : (tensor<64x64x3x3xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 192 + d1 * 3 + d2, d3), <1x1>, memref<12288x3xbf16, #system_memory>>>, !ttnn.device) -> tensor<1x1x576x64xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 576 + d1 * 576 + d2, d3), <1x1>, memref<18x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>

    // CHECK: "ttnn.conv2d"
    // CHECK-SAME: conv2d_config = #ttnn.conv2d_config<weights_dtype = bfp_bf8
    // CHECK-SAME: -> tensor<1x1x48400x64xbf16,
    %1 = "ttnn.conv2d"(%arg1, %0, %arg2) <{
      batch_size = 1 : i32,
      conv2d_config = #ttnn.conv2d_config<
        weights_dtype = bf16,
        deallocate_activation = false,
        reallocate_halo_output = false,
        act_block_h_override = 0,
        act_block_w_div = 1,
        reshard_if_not_optimal = false,
        override_sharding_config = false,
        transpose_shards = false,
        output_layout = tile,
        enable_act_double_buffer = false,
        enable_weights_double_buffer = false,
        in_place = false,
        enable_kernel_stride_folding = false>,
      conv2d_slice_config = #ttnn.conv2d_slice_config<l1_full, 0>,
      dilation = array<i32: 2, 2>,
      dtype = #ttcore.supportedDataTypes<bf16>,
      groups = 1 : i32,
      in_channels = 64 : i32,
      input_height = 224 : i32,
      input_width = 224 : i32,
      kernel_size = array<i32: 3, 3>,
      out_channels = 64 : i32,
      padding = array<i32: 0, 0, 0, 0>,
      stride = array<i32: 1, 1>
    }> : (tensor<1x1x50176x64xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 50176 + d1 * 50176 + d2, d3), <1x1>, memref<1568x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>, tensor<1x1x576x64xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 576 + d1 * 576 + d2, d3), <1x1>, memref<18x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>, !ttnn.device) -> tensor<1x1x48400x64xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 48416 + d1 * 48416 + d2, d3), <1x1>, memref<1513x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>

    // CHECK: return {{.*}} : tensor<1x1x48400x64xbf16,
    return %1 : tensor<1x1x48400x64xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 48416 + d1 * 48416 + d2, d3), <1x1>, memref<1513x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>>
  }
}
