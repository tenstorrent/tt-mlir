// RUN: ttmlir-opt --ttcore-register-device --ttnn-workaround --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t
#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 16384 + d1 * 128 + d2, d3), <1x1>, memref<16384x32xf32, #system_memory>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 4096 + d1 * 64 + d2, d3), <1x1>, memref<4096x32xf32, #system_memory>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 16384 + d1 * 128 + d2, d3), <1x1>, memref<512x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 4096 + d1 * 64 + d2, d3), <1x1>, memref<128x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
module attributes {} {
  func.func @forward(%arg0: tensor<1x128x128x32xf32, #ttnn_layout>) -> tensor<1x64x64x32xf32, #ttnn_layout1> {
    %0 = "ttnn.to_layout"(%arg0) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<1x128x128x32xf32, #ttnn_layout>) -> tensor<1x128x128x32xf32, #ttnn_layout2>
    // CHECK: "ttnn.to_layout"
    %1 = "ttnn.reshape"(%0) <{shape = [1 : i32, 1 : i32, 16384 : i32, 32 : i32]}> : (tensor<1x128x128x32xf32, #ttnn_layout2>) -> tensor<1x1x16384x32xf32, #ttnn_layout2>
    // Check that the input operand is transformed into the row major layout.
    // CHECK: %[[TO_LAYOUT_INPUT:.*]] = "ttnn.to_layout"
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK-SAME: layout = #ttnn.layout<row_major>
    // CHECK-SAME: memory_config = #ttnn.memory_config<#dram, <interleaved>>
    // CHECK-SAME: -> tensor<1x1x16384x32xbf16,
    // %3 = "ttnn.max_pool2d"(%2, %0) <{batch_size = 1 : si32, ceil_mode = false, channels = 32 : si32, dilation_height = 1 : si32, dilation_width = 1 : si32, input_height = 128 : si32, input_width = 128 : si32, kernel_height = 2 : si32, kernel_width = 2 : si32, padding_height = 0 : si32, padding_width = 0 : si32, stride_height = 2 : si32, stride_width = 2 : si32}> : (tensor<1x1x16384x32xf32, #ttnn_layout2>, !ttcore.device<#device>) -> tensor<1x1x4096x32xf32, #ttnn_layout3>
    %2 = "ttnn.max_pool2d"(%1) <{batch_size = 1 : si32, ceil_mode = false, channels = 32 : si32, input_height = 128 : si32, input_width = 128 : si32, kernel_size = array<i32: 2, 2>, stride = array<i32: 2, 2>, dilation = array<i32: 1, 1>, padding = array<i32: 0, 0>, in_place_halo = false}> : (tensor<1x1x16384x32xf32, #ttnn_layout2>) -> tensor<1x1x4096x32xf32, #ttnn_layout3>
    // CHECK-NEXT: %[[MAX_POOL_2D_OP:.*]] = "ttnn.max_pool2d"(%[[TO_LAYOUT_INPUT]])
    // Check that the output operand is transformed back into the tile and f32 data type.
    // CHECK-NEXT: %[[TO_LAYOUT_OUTPUT:.*]] = "ttnn.to_layout"(%[[MAX_POOL_2D_OP]])
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<f32>
    // CHECK-SAME: layout = #ttnn.layout<tile>
    // CHECK-SAME: memory_config = #ttnn.memory_config<#dram, <interleaved>>
    // CHECK-SAME: -> tensor<1x1x4096x32xf32
    %3 = "ttnn.view"(%2) <{shape = [1 : i32, 64 : i32, 64 : i32, 32 : i32]}> : (tensor<1x1x4096x32xf32, #ttnn_layout3>) -> tensor<1x64x64x32xf32, #ttnn_layout3>
    // CHECK-NEXT: ttnn.view
    %4 = "ttnn.to_layout"(%3) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#system_memory>}> : (tensor<1x64x64x32xf32, #ttnn_layout3>) -> tensor<1x64x64x32xf32, #ttnn_layout1>
    return %4 : tensor<1x64x64x32xf32, #ttnn_layout1>
  }
}
