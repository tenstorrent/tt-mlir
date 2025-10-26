// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 32 + d2, d3), <1x1>, memref<24x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 768 + d1 * 3 + d2, d3), <1x1>, memref<196608x3xbf16, #system_memory>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 320 + d1 * 32 + d2, d3), <1x1>, memref<30x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout4 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 64 + d1 * 8 + d2, d3), <1x1>, memref<192x256xbf16, #dram>, <interleaved>>

#device_compute_kernel_config = #ttnn.device_compute_kernel_config<
  math_fidelity = lofi,
  math_approx_mode = true,
  fp32_dest_acc_en = false,
  packer_l1_acc = false,
  dst_full_sync_en = false
>

#conv2d_config = #ttnn.conv2d_config<
  weights_dtype = bf16,
  deallocate_activation = false,
  reallocate_halo_output = true,
  act_block_h_override = 0,
  act_block_w_div = 1,
  reshard_if_not_optimal = false,
  override_sharding_config = false,
  shard_layout = height_sharded,
  core_grid = #ttnn.core_range_set<>,
  transpose_shards = true,
  output_layout = tile,
  enable_act_double_buffer = false,
  enable_weights_double_buffer = false,
  in_place = false
>

module attributes {} {
  func.func @forward(%arg0: tensor<3x8x8x256xbf16, #ttnn_layout>, %arg1: tensor<256x256x3x3xbf16, #ttnn_layout1>, %arg2: tensor<1x1x1x256xbf16, #ttnn_layout2>) -> tensor<3x10x10x256xbf16, #ttnn_layout3> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.to_layout"(%arg0) <{layout = #ttnn.layout<row_major>}> : (tensor<3x8x8x256xbf16, #ttnn_layout>) -> tensor<3x8x8x256xbf16, #ttnn_layout4>
    "ttnn.deallocate"(%arg0) <{force = false}> : (tensor<3x8x8x256xbf16, #ttnn_layout>) -> ()
    %2 = "ttnn.conv_transpose2d"(%1, %arg1, %arg2, %0)
            <{
              in_channels = 256 : i32,
              out_channels = 256 : i32,
              batch_size = 3 : i32,
              input_height = 8 : i32,
              input_width = 8 : i32,
              kernel_size = array<i32: 3, 3>,
              stride = array<i32: 1, 1>,
              padding = array<i32: 0, 0>,
              output_padding = array<i32: 0, 0>,
              dilation = array<i32: 1, 1>,
              groups = 1 : i32,
              conv2d_config = #conv2d_config,
              compute_config = #device_compute_kernel_config,
              dtype = #ttcore.supportedDataTypes<bf16>
            }> : (tensor<3x8x8x256xbf16, #ttnn_layout4>, tensor<256x256x3x3xbf16, #ttnn_layout1>, tensor<1x1x1x256xbf16, #ttnn_layout2>, !ttnn.device) -> tensor<1x1x300x256xbf16, #ttnn_layout3>
    "ttnn.deallocate"(%1) <{force = false}> : (tensor<3x8x8x256xbf16, #ttnn_layout4>) -> ()
    "ttnn.deallocate"(%arg2) <{force = false}> : (tensor<1x1x1x256xbf16, #ttnn_layout2>) -> ()
    "ttnn.deallocate"(%arg1) <{force = false}> : (tensor<256x256x3x3xbf16, #ttnn_layout1>) -> ()
    %3 = "ttnn.reshape"(%2) <{shape = [3 : i32, 10 : i32, 10 : i32, 256 : i32]}> : (tensor<1x1x300x256xbf16, #ttnn_layout3>) -> tensor<3x10x10x256xbf16, #ttnn_layout3>
    "ttnn.deallocate"(%2) <{force = false}> : (tensor<1x1x300x256xbf16, #ttnn_layout3>) -> ()
    return %3 : tensor<3x10x10x256xbf16, #ttnn_layout3>
  }
}
