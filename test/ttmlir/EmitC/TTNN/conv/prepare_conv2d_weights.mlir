// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %basename_t.ttnn
// RUN: ttmlir-opt --ttnn-backend-to-emitc-pipeline="target-dylib=true" %t.mlir > %t2.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir > %basename_t.cpp

#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 192 + d1 * 3 + d2, d3), <1x1>, memref<12288x3xbf16, #system_memory>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 576 + d1 * 576 + d2, d3), <1x1>, memref<18x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

func.func @prepare_conv2d_weights(%arg0: tensor<64x64x3x3xbf16, #ttnn_layout>) -> tensor<1x1x576x64xbf16, #ttnn_layout1> {
  %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
  %1 = "ttnn.prepare_conv2d_weights"(%arg0, %0)
          <{
              batch_size = 1 : i32,
              dilation = array<i32: 1, 1>,
              groups = 1 : i32,
              has_bias = true,
              in_channels = 64 : i32,
              input_height = 32 : i32,
              input_memory_config = #ttnn.memory_config<#dram, <interleaved>>,
              input_tensor_layout = #ttnn.layout<tile>,
              input_width = 32 : i32,
              kernel_size = array<i32: 3, 3>,
              out_channels = 64 : i32,
              padding = array<i32: 0, 0>,
              stride = array<i32: 1, 1>,
              weights_format = "OIHW"
          }> : (tensor<64x64x3x3xbf16, #ttnn_layout>, !ttnn.device) -> tensor<1x1x576x64xbf16, #ttnn_layout1>
  return %1 : tensor<1x1x576x64xbf16, #ttnn_layout1>
}
