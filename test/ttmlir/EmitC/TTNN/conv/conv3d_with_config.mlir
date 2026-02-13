// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttcore-mark-functions-as-forward -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t.mlir
// RUN: ttmlir-opt --ttnn-common-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3, d4) -> (d0 * 7168 + d1 * 896 + d2 * 32 + d3, d4), <1x1>, memref<224x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1, d2, d3, d4) -> (d0 * 3072 + d1 * 96 + d2 * 32 + d3, d4), <1x1>, memref<3072x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1, d2, d3, d4) -> (d0 * 32 + d1 * 32 + d2 * 32 + d3, d4), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1, d2, d3, d4) -> (d0 * 4992 + d1 * 832 + d2 * 32 + d3, d4), <1x1>, memref<156x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout4 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<27x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout5 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout6 = #ttnn.ttnn_layout<(d0, d1, d2, d3, d4) -> (d0 * 6272 + d1 * 784 + d2 * 28 + d3, d4), <1x1>, memref<6272x32xbf16, #dram>, <interleaved>>

#conv3d_config = #ttnn.conv3d_config<
  weights_dtype = bf16,
  t_out_block = 1,
  w_out_block = 1,
  h_out_block = 1,
  c_out_block = 32,
  c_in_block = 32,
  compute_with_storage_grid_size = #ttcore.grid<8x8>
>

func.func @conv3d_with_config(%arg0: tensor<1x8x28x28x32xbf16, #ttnn_layout>, %arg1: tensor<32x32x3x3x3xbf16, #ttnn_layout1>, %arg2: tensor<1x1x1x1x32xbf16, #ttnn_layout2>) -> tensor<1x6x26x26x32xbf16, #ttnn_layout3> {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.reshape"(%arg1) <{shape = [864 : i32, 32 : i32]}> : (tensor<32x32x3x3x3xbf16, #ttnn_layout1>) -> tensor<864x32xbf16, #ttnn_layout4>
  "ttnn.deallocate"(%arg1) <{force = false}> : (tensor<32x32x3x3x3xbf16, #ttnn_layout1>) -> ()
  %2 = "ttnn.reshape"(%arg2) <{shape = [1 : i32, 32 : i32]}> : (tensor<1x1x1x1x32xbf16, #ttnn_layout2>) -> tensor<1x32xbf16, #ttnn_layout5>
  "ttnn.deallocate"(%arg2) <{force = false}> : (tensor<1x1x1x1x32xbf16, #ttnn_layout2>) -> ()
  %3 = "ttnn.to_layout"(%arg0) <{layout = #ttnn.layout<row_major>}> : (tensor<1x8x28x28x32xbf16, #ttnn_layout>) -> tensor<1x8x28x28x32xbf16, #ttnn_layout6>
  "ttnn.deallocate"(%arg0) <{force = false}> : (tensor<1x8x28x28x32xbf16, #ttnn_layout>) -> ()
  %4 = "ttnn.conv3d"(%3, %1, %2, %0) <{
    batch_size = 1 : i32,
    conv3d_config = #conv3d_config,
    dtype = #ttcore.supportedDataTypes<bf16>,
    groups = 1 : i32,
    in_channels = 32 : i32,
    input_depth = 8 : i32,
    input_height = 28 : i32,
    input_width = 28 : i32,
    kernel_size = array<i32: 3, 3, 3>,
    out_channels = 32 : i32,
    padding = array<i32: 0, 0, 0>,
    padding_mode = "zeros",
    stride = array<i32: 1, 1, 1>
  }> : (tensor<1x8x28x28x32xbf16, #ttnn_layout6>, tensor<864x32xbf16, #ttnn_layout4>, tensor<1x32xbf16, #ttnn_layout5>, !ttnn.device) -> tensor<1x6x26x26x32xbf16, #ttnn_layout3>
  "ttnn.deallocate"(%3) <{force = false}> : (tensor<1x8x28x28x32xbf16, #ttnn_layout6>) -> ()
  "ttnn.deallocate"(%2) <{force = false}> : (tensor<1x32xbf16, #ttnn_layout5>) -> ()
  "ttnn.deallocate"(%1) <{force = false}> : (tensor<864x32xbf16, #ttnn_layout4>) -> ()
  return %4 : tensor<1x6x26x26x32xbf16, #ttnn_layout3>
}
