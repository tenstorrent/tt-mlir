// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t.mlir
// RUN: ttmlir-opt --ttnn-backend-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 24 + d1 * 3 + d2, d3), <1x1>, memref<384x3xbf16, #system_memory>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 96 + d1 * 96 + d2, d3), <1x1>, memref<3x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

func.func @prepare_conv2d_weights(%arg0: tensor<16x8x3x3xbf16, #ttnn_layout>) -> tensor<1x1x72x16xbf16, #ttnn_layout1> {
  %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.prepare_conv2d_weights"(%arg0, %0)
      <{
          batch_size = 3 : i32,
          dilation = array<i32: 1, 1>,
          groups = 1 : i32,
          has_bias = true,
          in_channels = 8 : i32,
          input_height = 32 : i32,
          input_memory_config = #ttnn.memory_config<#dram, <interleaved>>,
          input_tensor_layout = #ttnn.layout<tile>,
          input_width = 32 : i32,
          kernel_size = array<i32: 3, 3>,
          out_channels = 16 : i32,
          padding = array<i32: 0, 2, 1, 3>,
          stride = array<i32: 2, 2>,
          weights_format = "OIHW",
          input_dtype = #ttcore.supportedDataTypes<bf16>,
          output_dtype = #ttcore.supportedDataTypes<bf16>
      }> : (tensor<16x8x3x3xbf16, #ttnn_layout>, !ttnn.device) -> tensor<1x1x72x16xbf16, #ttnn_layout1>
  return %1 : tensor<1x1x72x16xbf16, #ttnn_layout1>
}
