// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t.mlir
// RUN: ttmlir-opt --ttnn-backend-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<32x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 48 + d1 * 3 + d2, d3), <1x1>, memref<3072x3xbf16, #system_memory>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 + d1 + d2, d3), <1x1>, memref<1x64xbf16, #system_memory>>
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 32 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout4 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 1024 + d2, d3), <1x1>, memref<4x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout5 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 576 + d1 * 576 + d2, d3), <1x1>, memref<3x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout6 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout7 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

#conv2d_config_prepare_conv2d_bias = #ttnn.conv2d_config<
  weights_dtype = bf16
>

func.func @conv2d_with_prepare_conv2d_weights_and_prepare_conv2d_bias(%arg0: tensor<1x32x32x64xbf16, #ttnn_layout>, %arg1: tensor<64x16x3x3xbf16, #ttnn_layout1>, %arg2: tensor<1x1x1x64xbf16, #ttnn_layout2>) -> tensor<1x8x4x64xbf16, #ttnn_layout3> {
  %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.reshape"(%arg0) <{shape = [1 : i32, 1 : i32, 1024 : i32, 64 : i32]}> : (tensor<1x32x32x64xbf16, #ttnn_layout>) -> tensor<1x1x1024x64xbf16, #ttnn_layout4>
  "ttnn.deallocate"(%arg0) <{force = false}> : (tensor<1x32x32x64xbf16, #ttnn_layout>) -> ()
  %2 = "ttnn.prepare_conv2d_weights"(%arg1, %0)
        <{
          batch_size = 1 : i32,
          dilation = array<i32: 2, 4>,
          groups = 4 : i32,
          has_bias = true,
          in_channels = 64 : i32,
          input_height = 32 : i32,
          input_memory_config = #ttnn.memory_config<#dram, <interleaved>>,
          input_tensor_layout = #ttnn.layout<tile>,
          input_width = 32 : i32,
          kernel_size = array<i32: 3, 3>,
          out_channels = 64 : i32,
          padding = array<i32: 2, 4>,
          stride = array<i32: 4, 8>,
          weights_format = "OIHW",
          input_dtype = #ttcore.supportedDataTypes<bf16>,
          dtype = #ttcore.supportedDataTypes<bf16>
        }> : (tensor<64x16x3x3xbf16, #ttnn_layout1>, !ttnn.device) -> tensor<1x1x576x64xbf16, #ttnn_layout5>
  "ttnn.deallocate"(%arg1) <{force = false}> : (tensor<64x16x3x3xbf16, #ttnn_layout1>) -> ()
  %3 = "ttnn.prepare_conv2d_bias"(%arg2, %0)
        <{
          batch_size = 1 : i32,
          conv2d_config = #conv2d_config_prepare_conv2d_bias,
          dilation = array<i32: 2, 4>,
          groups = 4 : i32,
          in_channels = 64 : i32,
          input_height = 32 : i32,
          input_memory_config = #ttnn.memory_config<#dram, <interleaved>>,
          input_tensor_layout = #ttnn.layout<tile>,
          input_width = 32 : i32,
          kernel_size = array<i32: 3, 3>,
          out_channels = 64 : i32,
          padding = array<i32: 2, 4>,
          stride = array<i32: 4, 8>,
          input_dtype = #ttcore.supportedDataTypes<bf16>,
          dtype = #ttcore.supportedDataTypes<bf16>
        }> : (tensor<1x1x1x64xbf16, #ttnn_layout2>, !ttnn.device) -> tensor<1x1x1x64xbf16, #ttnn_layout6>
  "ttnn.deallocate"(%arg2) <{force = false}> : (tensor<1x1x1x64xbf16, #ttnn_layout2>) -> ()
  %4 = "ttnn.conv2d"(%1, %2, %3, %0) <{batch_size = 1 : i32, dilation = array<i32: 2, 4>, groups = 4 : i32, in_channels = 64 : i32, input_height = 32 : i32, input_width = 32 : i32, kernel_size = array<i32: 3, 3>, out_channels = 64 : i32, padding = array<i32: 2, 4>, stride = array<i32: 4, 8>, dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<1x1x1024x64xbf16, #ttnn_layout4>, tensor<1x1x576x64xbf16, #ttnn_layout5>, tensor<1x1x1x64xbf16, #ttnn_layout6>, !ttnn.device) -> tensor<1x1x32x64xbf16, #ttnn_layout7>
  "ttnn.deallocate"(%3) <{force = false}> : (tensor<1x1x1x64xbf16, #ttnn_layout6>) -> ()
  "ttnn.deallocate"(%2) <{force = false}> : (tensor<1x1x576x64xbf16, #ttnn_layout5>) -> ()
  "ttnn.deallocate"(%1) <{force = false}> : (tensor<1x1x1024x64xbf16, #ttnn_layout4>) -> ()
  %5 = "ttnn.reshape"(%4) <{shape = [1 : i32, 8 : i32, 4 : i32, 64 : i32]}> : (tensor<1x1x32x64xbf16, #ttnn_layout7>) -> tensor<1x8x4x64xbf16, #ttnn_layout3>
  "ttnn.deallocate"(%4) <{force = false}> : (tensor<1x1x32x64xbf16, #ttnn_layout7>) -> ()
  return %5 : tensor<1x8x4x64xbf16, #ttnn_layout3>
}
