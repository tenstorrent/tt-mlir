// RUN: ttmlir-opt --ttcore-register-device --ttnn-prepare-conv3d-weights %s | FileCheck %s

// Verifies that the post-optimizer pass materializes a PrepareConv3dWeightsOp
// before each Conv3dOp, picking c_in_block from the optimizer's
// Conv3dConfigAttr when present (falling back to TILE_WIDTH=32 otherwise).
//
// Shapes: in_channels=128, out_channels=32, kernel=3x3x3. c_in_aligned=128,
// so c_in_block in {32, 64, 128} all satisfy the divisibility check.

#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout_in = #ttnn.ttnn_layout<(d0, d1, d2, d3, d4) -> (d0 * 6272 + d1 * 784 + d2 * 28 + d3, d4), <1x1>, memref<6272x128xbf16, #dram>, <interleaved>>
#ttnn_layout_weight_raw = #ttnn.ttnn_layout<(d0, d1, d2, d3, d4) -> (d0 * 1152 + d1 * 9 + d2 * 3 + d3, d4), <1x1>, memref<4608x3xbf16, #system_memory>>
#ttnn_layout_out = #ttnn.ttnn_layout<(d0, d1, d2, d3, d4) -> (d0 * 4056 + d1 * 676 + d2 * 26 + d3, d4), <1x1>, memref<4056x32xbf16, #dram>, <interleaved>>

module {
  // Case 1: optimizer chose c_in_block=64. The pass must insert a
  // PrepareConv3dWeightsOp with that c_in_block.
  // CHECK-LABEL: @conv3d_with_chosen_c_in_block
  func.func @conv3d_with_chosen_c_in_block(
      %arg0: tensor<1x8x28x28x128xbf16, #ttnn_layout_in>,
      %arg1: tensor<32x128x3x3x3xbf16, #ttnn_layout_weight_raw>)
      -> tensor<1x6x26x26x32xbf16, #ttnn_layout_out> {
    %dev = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: "ttnn.prepare_conv3d_weights"
    // CHECK-SAME: c_in_block = 64 : i32
    // CHECK: "ttnn.conv3d"
    %out = "ttnn.conv3d"(%arg0, %arg1, %dev)
        <{batch_size = 1 : i32,
          conv3d_config = #ttnn.conv3d_config<c_in_block = 64>,
          dtype = #ttcore.supportedDataTypes<bf16>,
          groups = 1 : i32,
          in_channels = 128 : i32,
          input_depth = 8 : i32, input_height = 28 : i32, input_width = 28 : i32,
          kernel_size = array<i32: 3, 3, 3>,
          out_channels = 32 : i32,
          padding = array<i32: 0, 0, 0>,
          padding_mode = "zeros",
          stride = array<i32: 1, 1, 1>}>
        : (tensor<1x8x28x28x128xbf16, #ttnn_layout_in>,
           tensor<32x128x3x3x3xbf16, #ttnn_layout_weight_raw>,
           !ttnn.device)
        -> tensor<1x6x26x26x32xbf16, #ttnn_layout_out>
    return %out : tensor<1x6x26x26x32xbf16, #ttnn_layout_out>
  }

  // Case 2: no conv3d_config on the op. The pass must insert a prepare op
  // with c_in_block = TILE_WIDTH = 32 (tt-metal default).
  // CHECK-LABEL: @conv3d_without_config
  func.func @conv3d_without_config(
      %arg0: tensor<1x8x28x28x128xbf16, #ttnn_layout_in>,
      %arg1: tensor<32x128x3x3x3xbf16, #ttnn_layout_weight_raw>)
      -> tensor<1x6x26x26x32xbf16, #ttnn_layout_out> {
    %dev = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: "ttnn.prepare_conv3d_weights"
    // CHECK-SAME: c_in_block = 32 : i32
    // CHECK: "ttnn.conv3d"
    %out = "ttnn.conv3d"(%arg0, %arg1, %dev)
        <{batch_size = 1 : i32,
          dtype = #ttcore.supportedDataTypes<bf16>,
          groups = 1 : i32,
          in_channels = 128 : i32,
          input_depth = 8 : i32, input_height = 28 : i32, input_width = 28 : i32,
          kernel_size = array<i32: 3, 3, 3>,
          out_channels = 32 : i32,
          padding = array<i32: 0, 0, 0>,
          padding_mode = "zeros",
          stride = array<i32: 1, 1, 1>}>
        : (tensor<1x8x28x28x128xbf16, #ttnn_layout_in>,
           tensor<32x128x3x3x3xbf16, #ttnn_layout_weight_raw>,
           !ttnn.device)
        -> tensor<1x6x26x26x32xbf16, #ttnn_layout_out>
    return %out : tensor<1x6x26x26x32xbf16, #ttnn_layout_out>
  }

  // Case 3: conv3d_config attached but c_in_block field is nullopt. The
  // pass must fall back to TILE_WIDTH = 32.
  // CHECK-LABEL: @conv3d_config_without_c_in_block
  func.func @conv3d_config_without_c_in_block(
      %arg0: tensor<1x8x28x28x128xbf16, #ttnn_layout_in>,
      %arg1: tensor<32x128x3x3x3xbf16, #ttnn_layout_weight_raw>)
      -> tensor<1x6x26x26x32xbf16, #ttnn_layout_out> {
    %dev = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: "ttnn.prepare_conv3d_weights"
    // CHECK-SAME: c_in_block = 32 : i32
    // CHECK: "ttnn.conv3d"
    %out = "ttnn.conv3d"(%arg0, %arg1, %dev)
        <{batch_size = 1 : i32,
          conv3d_config = #ttnn.conv3d_config<h_out_block = 2>,
          dtype = #ttcore.supportedDataTypes<bf16>,
          groups = 1 : i32,
          in_channels = 128 : i32,
          input_depth = 8 : i32, input_height = 28 : i32, input_width = 28 : i32,
          kernel_size = array<i32: 3, 3, 3>,
          out_channels = 32 : i32,
          padding = array<i32: 0, 0, 0>,
          padding_mode = "zeros",
          stride = array<i32: 1, 1, 1>}>
        : (tensor<1x8x28x28x128xbf16, #ttnn_layout_in>,
           tensor<32x128x3x3x3xbf16, #ttnn_layout_weight_raw>,
           !ttnn.device)
        -> tensor<1x6x26x26x32xbf16, #ttnn_layout_out>
    return %out : tensor<1x6x26x26x32xbf16, #ttnn_layout_out>
  }
}
