// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t.mlir
// RUN: ttmlir-opt --ttnn-backend-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<1024x64xbf16, #system_memory>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 192 + d1 * 3 + d2, d3), <1x1>, memref<12288x3xbf16, #system_memory>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 + d1 + d2, d3), <1x1>, memref<1x64xbf16, #system_memory>>
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 900 + d1 * 30 + d2, d3), <1x1>, memref<900x64xbf16, #system_memory>>
#ttnn_layout4 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 900 + d1 * 30 + d2, d3), <1x1>, memref<29x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout5 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 900 + d1 * 30 + d2, d3), <1x1>, memref<29x2x!ttcore.tile<32x32, bf16>, #system_memory>>

#conv2d_config = #ttnn.conv2d_config<
  weights_dtype = bf16,
  activation = "",
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

func.func @conv2d_conv2dconfig(%arg0: tensor<1x1x1024x64xbf16, #ttnn_layout>, %arg1: tensor<64x64x3x3xbf16, #ttnn_layout1>, %arg2: tensor<1x1x1x64xbf16, #ttnn_layout2>) -> tensor<1x30x30x64xbf16, #ttnn_layout3> {
  %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.conv2d"(%arg0, %arg1, %arg2, %0)
      <{
        in_channels = 64 : i32,
        out_channels = 64 : i32,
        batch_size = 1 : i32,
        input_height = 32 : i32,
        input_width = 32 : i32,
        kernel_size = array<i32: 3, 3>,
        stride = array<i32: 1, 1>,
        padding = array<i32: 0, 0>,
        dilation = array<i32: 1, 1>,
        groups = 1 : i32,
        conv2d_config = #conv2d_config,
        dtype = #ttcore.supportedDataTypes<bf16>
      }> : (tensor<1x1x1024x64xbf16, #ttnn_layout>, tensor<64x64x3x3xbf16, #ttnn_layout1>, tensor<1x1x1x64xbf16, #ttnn_layout2>, !ttnn.device) -> tensor<1x1x900x64xbf16, #ttnn_layout4>
  %2 = "ttnn.reshape"(%1) <{shape = [1 : i32, 30 : i32, 30 : i32, 64 : i32]}> : (tensor<1x1x900x64xbf16, #ttnn_layout4>) -> tensor<1x30x30x64xbf16, #ttnn_layout4>
  "ttnn.deallocate"(%1) <{force = false}> : (tensor<1x1x900x64xbf16, #ttnn_layout4>) -> ()
  %3 = "ttnn.from_device"(%2) : (tensor<1x30x30x64xbf16, #ttnn_layout4>) -> tensor<1x30x30x64xbf16, #ttnn_layout5>
  "ttnn.deallocate"(%2) <{force = false}> : (tensor<1x30x30x64xbf16, #ttnn_layout4>) -> ()
  %4 = "ttnn.to_layout"(%3) <{layout = #ttnn.layout<row_major>}> : (tensor<1x30x30x64xbf16, #ttnn_layout5>) -> tensor<1x30x30x64xbf16, #ttnn_layout3>
  "ttnn.deallocate"(%3) <{force = false}> : (tensor<1x30x30x64xbf16, #ttnn_layout5>) -> ()
  return %4 : tensor<1x30x30x64xbf16, #ttnn_layout3>
}
