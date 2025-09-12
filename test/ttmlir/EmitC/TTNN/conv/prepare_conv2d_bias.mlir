// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t.mlir
// RUN: ttmlir-opt --ttnn-backend-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>

#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0, d1, d2, d3), <1x1>, memref<1x64xbf16, #system_memory>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0, d1, d2, d3), <1x1>, memref<1x64xf32, #system_memory>>
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

#conv2d_config_bf16 = #ttnn.conv2d_config<weights_dtype = bf16>
#conv2d_config_f32 = #ttnn.conv2d_config<weights_dtype = f32>

module {
  func.func @prepare_conv2d_bias_bf16(%arg0: tensor<1x1x1x64xbf16, #ttnn_layout>) -> tensor<1x1x1x64xbf16, #ttnn_layout1> {
    %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.prepare_conv2d_bias"(%arg0, %0)
          <{
            batch_size = 16 : i32,
            dilation = array<i32: 1, 1>,
            groups = 1 : i32,
            in_channels = 64 : i32,
            input_height = 32 : i32,
            input_memory_config = #ttnn.memory_config<#dram, <interleaved>>,
            input_tensor_layout = #ttnn.layout<tile>,
            input_width = 32 : i32,
            kernel_size = array<i32: 3, 3>,
            out_channels = 64 : i32,
            padding = array<i32: 0, 0>,
            stride = array<i32: 1, 1>,
            conv2d_config = #conv2d_config_bf16,
            input_dtype = #ttcore.supportedDataTypes<bf16>,
            output_dtype = #ttcore.supportedDataTypes<bf16>
          }> : (tensor<1x1x1x64xbf16, #ttnn_layout>, !ttnn.device) -> tensor<1x1x1x64xbf16, #ttnn_layout1>
    "ttnn.deallocate"(%arg0) <{force = false}> : (tensor<1x1x1x64xbf16, #ttnn_layout>) -> ()
    return %1 : tensor<1x1x1x64xbf16, #ttnn_layout1>
  }

  func.func @prepare_conv2d_bias_f32(%arg0: tensor<1x1x1x64xf32, #ttnn_layout2>) -> tensor<1x1x1x64xf32, #ttnn_layout3> {
    %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.prepare_conv2d_bias"(%arg0, %0)
          <{
            batch_size = 16 : i32,
            dilation = array<i32: 1, 1>,
            groups = 1 : i32,
            in_channels = 64 : i32,
            input_height = 32 : i32,
            input_memory_config = #ttnn.memory_config<#dram, <interleaved>>,
            input_tensor_layout = #ttnn.layout<tile>,
            input_width = 32 : i32,
            kernel_size = array<i32: 3, 3>,
            out_channels = 64 : i32,
            padding = array<i32: 0, 0>,
            stride = array<i32: 1, 1>,
            conv2d_config = #conv2d_config_f32,
            input_dtype = #ttcore.supportedDataTypes<f32>,
            output_dtype = #ttcore.supportedDataTypes<f32>
          }> : (tensor<1x1x1x64xf32, #ttnn_layout2>, !ttnn.device) -> tensor<1x1x1x64xf32, #ttnn_layout3>
    "ttnn.deallocate"(%arg0) <{force = false}> : (tensor<1x1x1x64xf32, #ttnn_layout2>) -> ()
    return %1 : tensor<1x1x1x64xf32, #ttnn_layout3>
  }
}
