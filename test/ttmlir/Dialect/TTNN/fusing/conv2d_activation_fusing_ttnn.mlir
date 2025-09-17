// RUN: ttmlir-opt --ttcore-register-device --ttnn-fusing -o %t %s
// RUN: FileCheck %s --input-file=%t

// These test are written in TTNN dialect because TTIR is not flexible enough to represent all the patterns we want to test.

#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<32x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 192 + d1 * 3 + d2, d3), <1x1>, memref<12288x3xbf16, #system_memory>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 960 + d1 * 32 + d2, d3), <1x1>, memref<30x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout4 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 1024 + d2, d3), <1x1>, memref<32x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout5 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 928 + d1 * 928 + d2, d3), <1x1>, memref<29x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
module {
  // Here we want to test that we cant fuse relu into conv2d because conv2d already has activation.

  // CHECK-LABEL: func.func @conv2d_relu_chain
  func.func @conv2d_relu_chain(%arg0: tensor<1x32x32x64xbf16, #ttnn_layout>, %arg1: tensor<64x64x3x3xbf16, #ttnn_layout1>, %arg2: tensor<1x1x1x64xbf16, #ttnn_layout2>) -> tensor<1x30x30x64xbf16, #ttnn_layout3> {
    // CHECK: "ttnn.conv2d"
    // CHECK-SAME: activation = <op_type = relu>

    %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.reshape"(%arg0) <{shape = [1 : i32, 1 : i32, 1024 : i32, 64 : i32]}> : (tensor<1x32x32x64xbf16, #ttnn_layout>) -> tensor<1x1x1024x64xbf16, #ttnn_layout4>
    %2 = "ttnn.conv2d"(%1, %arg1, %arg2, %0)
          <{
            batch_size = 1 : i32,
            conv2d_config = #ttnn.conv2d_config<weights_dtype = bf16, activation = #ttnn.unary_with_param<op_type = relu>, deallocate_activation = false, reallocate_halo_output = false, act_block_h_override = 0, act_block_w_div = 1, reshard_if_not_optimal = false, override_sharding_config = false, shard_layout = height_sharded, transpose_shards = false, output_layout = tile, enable_act_double_buffer = false, enable_weights_double_buffer = false>,
            dilation = array<i32: 1, 1>,
            groups = 1 : i32,
            in_channels = 64 : i32,
            input_height = 32 : i32,
            input_width = 32 : i32,
            kernel_size = array<i32: 3, 3>,
            out_channels = 64 : i32,
            padding = array<i32: 0, 0>,
            stride = array<i32: 1, 1>,
            dtype = #ttcore.supportedDataTypes<bf16>
          }> : (tensor<1x1x1024x64xbf16, #ttnn_layout4>, tensor<64x64x3x3xbf16, #ttnn_layout1>, tensor<1x1x1x64xbf16, #ttnn_layout2>, !ttnn.device) -> tensor<1x1x900x64xbf16, #ttnn_layout5>
    %3 = "ttnn.reshape"(%2) <{shape = [1 : i32, 30 : i32, 30 : i32, 64 : i32]}> : (tensor<1x1x900x64xbf16, #ttnn_layout5>) -> tensor<1x30x30x64xbf16, #ttnn_layout3>

    // CHECK: "ttnn.relu"
    %4 = "ttnn.relu"(%3) : (tensor<1x30x30x64xbf16, #ttnn_layout3>) -> tensor<1x30x30x64xbf16, #ttnn_layout3>
    return %4 : tensor<1x30x30x64xbf16, #ttnn_layout3>
  }

  // Test that we can fuse conv2d -> relu into conv2d. This test differs from rest of the tests because in other tests we add flattening to conv2d which adds reshapes before and after conv2d.

  // CHECK-LABEL: func.func @conv2d_with_relu_after
  func.func @conv2d_with_relu_after(%arg0: tensor<1x32x32x64xbf16, #ttnn_layout>, %arg1: tensor<64x64x3x3xbf16, #ttnn_layout1>, %arg2: tensor<1x1x1x64xbf16, #ttnn_layout2>) -> tensor<1x30x30x64xbf16, #ttnn_layout3> {
    // CHECK: "ttnn.conv2d"
    // CHECK-SAME: activation = <op_type = relu>

    %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.reshape"(%arg0) <{shape = [1 : i32, 1 : i32, 1024 : i32, 64 : i32]}> : (tensor<1x32x32x64xbf16, #ttnn_layout>) -> tensor<1x1x1024x64xbf16, #ttnn_layout4>
    %2 = "ttnn.conv2d"(%1, %arg1, %arg2, %0)
          <{
            batch_size = 1 : i32,
            dilation = array<i32: 1, 1>,
            groups = 1 : i32,
            in_channels = 64 : i32,
            input_height = 32 : i32,
            input_width = 32 : i32,
            kernel_size = array<i32: 3, 3>,
            out_channels = 64 : i32,
            padding = array<i32: 0, 0>,
            stride = array<i32: 1, 1>,
            dtype = #ttcore.supportedDataTypes<bf16>
          }> : (tensor<1x1x1024x64xbf16, #ttnn_layout4>, tensor<64x64x3x3xbf16, #ttnn_layout1>, tensor<1x1x1x64xbf16, #ttnn_layout2>, !ttnn.device) -> tensor<1x1x900x64xbf16, #ttnn_layout5>

    // CHECK-NOT: "ttnn.relu"
    %3 = "ttnn.relu"(%2) : (tensor<1x1x900x64xbf16, #ttnn_layout5>) -> tensor<1x1x900x64xbf16, #ttnn_layout5>
    %4 = "ttnn.reshape"(%3) <{shape = [1 : i32, 30 : i32, 30 : i32, 64 : i32]}> : (tensor<1x1x900x64xbf16, #ttnn_layout5>) -> tensor<1x30x30x64xbf16, #ttnn_layout3>
    return %4 : tensor<1x30x30x64xbf16, #ttnn_layout3>
  }
}
