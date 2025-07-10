// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %basename_t.ttnn
// RUN: ttmlir-opt --ttnn-tuplify-tensors --convert-ttnn-to-emitc %t.mlir > %t2.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir > %basename_t.cpp

#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<32x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 48 + d1 * 3 + d2, d3), <1x1>, memref<3072x3xbf16, #system_memory>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 576 + d1 * 576 + d2, d3), <1x1>, memref<18x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout4 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 32 + d2, d3), <1x1>, memref<8x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout5 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 1024 + d2, d3), <1x1>, memref<32x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout6 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 1024 + d2, d3), <1x1>, memref<32x2x!ttcore.tile<32x32, bf16>, #system_memory>>
#ttnn_layout7 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 1024 + d2, d3), <1x1>, memref<1024x64xbf16, #system_memory>>
#ttnn_layout8 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 1024 + d2, d3), <1x1>, memref<1024x64xbf16, #dram>, <interleaved>>

func.func @conv2d_with_prepare_conv2d_weights(%arg0: tensor<1x32x32x64xbf16, #ttnn_layout>, %arg1: tensor<64x16x3x3xbf16, #ttnn_layout1>, %arg2: tensor<1x1x1x64xbf16, #ttnn_layout3>) -> tensor<1x8x4x64xbf16, #ttnn_layout4> {
  %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.reshape"(%arg0) <{shape = [1 : i32, 1 : i32, 1024 : i32, 64 : i32]}> : (tensor<1x32x32x64xbf16, #ttnn_layout>) -> tensor<1x1x1024x64xbf16, #ttnn_layout5>
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
            output_dtype = #ttcore.supportedDataTypes<bf16>
          }> : (tensor<64x16x3x3xbf16, #ttnn_layout1>, !ttnn.device) -> tensor<1x1x576x64xbf16, #ttnn_layout2>
  "ttnn.deallocate"(%arg1) <{force = false}> : (tensor<64x16x3x3xbf16, #ttnn_layout1>) -> ()
  %3 = "ttnn.from_device"(%1) : (tensor<1x1x1024x64xbf16, #ttnn_layout5>) -> tensor<1x1x1024x64xbf16, #ttnn_layout6>
  "ttnn.deallocate"(%1) <{force = false}> : (tensor<1x1x1024x64xbf16, #ttnn_layout5>) -> ()
  %4 = "ttnn.to_layout"(%3) <{layout = #ttnn.layout<row_major>}> : (tensor<1x1x1024x64xbf16, #ttnn_layout6>) -> tensor<1x1x1024x64xbf16, #ttnn_layout7>
  "ttnn.deallocate"(%3) <{force = false}> : (tensor<1x1x1024x64xbf16, #ttnn_layout6>) -> ()
  %5 = "ttnn.to_device"(%4, %0) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<1x1x1024x64xbf16, #ttnn_layout7>, !ttnn.device) -> tensor<1x1x1024x64xbf16, #ttnn_layout8>
  "ttnn.deallocate"(%4) <{force = false}> : (tensor<1x1x1024x64xbf16, #ttnn_layout7>) -> ()
  %6 = "ttnn.conv2d"(%5, %2, %arg2, %0)
        <{
          batch_size = 1 : i32,
          dilation = array<i32: 2, 4>,
          groups = 4 : i32,
          in_channels = 64 : i32,
          input_height = 32 : i32,
          input_width = 32 : i32,
          kernel_size = array<i32: 3, 3>,
          out_channels = 64 : i32,
          padding = array<i32: 2, 4>,
          stride = array<i32: 4, 8>,
          output_dtype = #ttcore.supportedDataTypes<bf16>
        }> : (tensor<1x1x1024x64xbf16, #ttnn_layout8>, tensor<1x1x576x64xbf16, #ttnn_layout2>, tensor<1x1x1x64xbf16, #ttnn_layout3>, !ttnn.device) -> tensor<1x1x32x64xbf16, #ttnn_layout4>
  "ttnn.deallocate"(%5) <{force = false}> : (tensor<1x1x1024x64xbf16, #ttnn_layout8>) -> ()
  "ttnn.deallocate"(%2) <{force = false}> : (tensor<1x1x576x64xbf16, #ttnn_layout2>) -> ()
  "ttnn.deallocate"(%arg2) <{force = false}> : (tensor<1x1x1x64xbf16, #ttnn_layout3>) -> ()
  %7 = "ttnn.reshape"(%6) <{shape = [1 : i32, 8 : i32, 4 : i32, 64 : i32]}> : (tensor<1x1x32x64xbf16, #ttnn_layout4>) -> tensor<1x8x4x64xbf16, #ttnn_layout4>
  "ttnn.deallocate"(%6) <{force = false}> : (tensor<1x1x32x64xbf16, #ttnn_layout4>) -> ()
  return %7 : tensor<1x8x4x64xbf16, #ttnn_layout4>
}
