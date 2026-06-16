// RUN: ttmlir-opt --ttcore-register-device --ttnn-prepare-conv3d-weights %s | FileCheck %s

// Verifies that the post-optimizer pass materializes a PrepareConv3dWeightsOp
// before each Conv3dOp, reading c_in_block straight from the op's
// Conv3dConfigAttr. The pass does not reason about tt-metal defaults and does
// not modify the config: TTIRToTTNN already attaches a complete config (and the
// optimizer only refines it), so the config simply passes through unchanged
// while the prepare op is created with the c_in_block it carries.
//
// Shapes: in_channels=128, out_channels=32, kernel=3x3x3. c_in_aligned=128,
// so c_in_block in {32, 64, 128} all satisfy the divisibility check.

#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout_in = #ttnn.ttnn_layout<(d0, d1, d2, d3, d4) -> (d0 * 6272 + d1 * 784 + d2 * 28 + d3, d4), <1x1>, memref<6272x128xbf16, #dram>, <interleaved>>
#ttnn_layout_weight_raw = #ttnn.ttnn_layout<(d0, d1, d2, d3, d4) -> (d0 * 1152 + d1 * 9 + d2 * 3 + d3, d4), <1x1>, memref<4608x3xbf16, #system_memory>>
#ttnn_layout_out = #ttnn.ttnn_layout<(d0, d1, d2, d3, d4) -> (d0 * 4056 + d1 * 676 + d2 * 26 + d3, d4), <1x1>, memref<4056x32xbf16, #dram>, <interleaved>>

module {
  // c_in_block = 64: the prepare op must be created with that value, and the
  // op's config must pass through unchanged.
  // CHECK-LABEL: @conv3d_c_in_block_64
  func.func @conv3d_c_in_block_64(
      %arg0: tensor<1x8x28x28x128xbf16, #ttnn_layout_in>,
      %arg1: tensor<32x128x3x3x3xbf16, #ttnn_layout_weight_raw>)
      -> tensor<1x6x26x26x32xbf16, #ttnn_layout_out> {
    %dev = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: "ttnn.prepare_conv3d_weights"
    // CHECK-SAME: c_in_block = 64 : i32
    // CHECK: "ttnn.conv3d"
    // CHECK-SAME: conv3d_config = #ttnn.conv3d_config<
    // CHECK-SAME: c_in_block = 64
    %out = "ttnn.conv3d"(%arg0, %arg1, %dev)
        <{batch_size = 1 : i32,
          conv3d_config = #ttnn.conv3d_config<t_out_block = 1, w_out_block = 1, h_out_block = 1, c_out_block = 32, c_in_block = 64, compute_with_storage_grid_size = #ttcore.grid<8x8>>,
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

  // c_in_block = 32: the pass reads whatever the config carries, so the prepare
  // op must use 32 here.
  // CHECK-LABEL: @conv3d_c_in_block_32
  func.func @conv3d_c_in_block_32(
      %arg0: tensor<1x8x28x28x128xbf16, #ttnn_layout_in>,
      %arg1: tensor<32x128x3x3x3xbf16, #ttnn_layout_weight_raw>)
      -> tensor<1x6x26x26x32xbf16, #ttnn_layout_out> {
    %dev = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: "ttnn.prepare_conv3d_weights"
    // CHECK-SAME: c_in_block = 32 : i32
    // CHECK: "ttnn.conv3d"
    // CHECK-SAME: conv3d_config = #ttnn.conv3d_config<
    // CHECK-SAME: c_in_block = 32
    %out = "ttnn.conv3d"(%arg0, %arg1, %dev)
        <{batch_size = 1 : i32,
          conv3d_config = #ttnn.conv3d_config<t_out_block = 1, w_out_block = 1, h_out_block = 1, c_out_block = 32, c_in_block = 32, compute_with_storage_grid_size = #ttcore.grid<8x8>>,
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
