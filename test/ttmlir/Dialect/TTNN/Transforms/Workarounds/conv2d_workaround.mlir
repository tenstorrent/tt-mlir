// RUN: ttmlir-opt --ttnn-workaround %s | FileCheck %s

#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<32x2x!tt.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 192 + d1 * 3 + d2, d3), <1x1>, memref<12288x3xbf16, #system_memory>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x2x!tt.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 960 + d1 * 32 + d2, d3), <1x1>, memref<30x2x!tt.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout4 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 1024 + d2, d3), <1x1>, memref<32x2x!tt.tile<32x32, bf16>, #dram>, <interleaved>>
module attributes {} {
  tt.device @default_device = <workerGrid = #tt.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5] -> (0, 0, (((d0 * s1) * (s2 * s3) + d1 * (s2 * s3) + d2) floordiv s4) mod 12, ((d0 * s1) * (s2 * s3) + d1 * (s2 * s3) + d2) floordiv (s4 * 12) + ((d0 * s1) * (s2 * s3) + d1 * (s2 * s3) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
  func.func @conv2d_with_bias(%arg0: tensor<1x32x32x64xbf16, #ttnn_layout>, %arg1: tensor<64x64x3x3xbf16, #ttnn_layout1>, %arg2: tensor<1x1x1x64xbf16, #ttnn_layout2>) -> tensor<1x30x30x64xbf16, #ttnn_layout3> {
    // CHECK: %[[DEVICE_OP:.*]] = "ttnn.get_device"
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.empty"(%0) <{dtype = #tt.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <<30x2>>, <interleaved>>, shape = #ttnn.shape<1x30x30x64>}> : (!ttnn.device) -> tensor<1x30x30x64xbf16, #ttnn_layout3>
    %2 = "ttnn.reshape"(%arg0) <{shape = [1 : i32, 1 : i32, 1024 : i32, 64 : i32]}> : (tensor<1x32x32x64xbf16, #ttnn_layout>) -> tensor<1x1x1024x64xbf16, #ttnn_layout4>
    // CHECK: %[[TO_LAYOUT_INPUT:.*]] = "ttnn.to_layout"
    // CHECK-SAME: dtype = #tt.supportedDataTypes<bf16>
    // CHECK-SAME: layout = #ttnn.layout<row_major>
    %3 = "ttnn.conv2d"(%2, %arg1, %arg2, %0) <{batch_size = 1 : i32, dilation = array<i32: 1, 1>, groups = 1 : i32, in_channels = 64 : i32, input_height = 32 : i32, input_width = 32 : i32, kernel_size = array<i32: 3, 3>, out_channels = 64 : i32, padding = array<i32: 0, 0>, stride = array<i32: 1, 1>}> : (tensor<1x1x1024x64xbf16, #ttnn_layout4>, tensor<64x64x3x3xbf16, #ttnn_layout1>, tensor<1x1x1x64xbf16, #ttnn_layout2>, !ttnn.device) -> tensor<1x1x900x64xbf16, #ttnn_layout3>
    // CHECK-NEXT: %[[CONV2D_RESULT:.*]] = "ttnn.conv2d"(%[[TO_LAYOUT_INPUT]], %[[CONV2D_WEIGHTS:.*]], %[[CONV2D_BIAS:.*]], %[[DEVICE_OP]])
    // CHECK-NEXT: %[[TO_LAYOUT_OUTPUT:.*]] = "ttnn.to_layout"(%[[CONV2D_RESULT]], %[[DEVICE_OP]])
    // CHECK-SAME: dtype = #tt.supportedDataTypes<bf16>
    // CHECK-SAME: layout = #ttnn.layout<tile>
    %4 = "ttnn.reshape"(%3) <{shape = [1 : i32, 30 : i32, 30 : i32, 64 : i32]}> : (tensor<1x1x900x64xbf16, #ttnn_layout3>) -> tensor<1x30x30x64xbf16, #ttnn_layout3>
    return %4 : tensor<1x30x30x64xbf16, #ttnn_layout3>
  }

  func.func @conv_transpose2d_with_bias(%arg0: tensor<1x32x32x64xbf16, #ttnn_layout>, %arg1: tensor<64x64x3x3xbf16, #ttnn_layout1>, %arg2: tensor<1x1x1x64xbf16, #ttnn_layout2>) -> tensor<1x30x30x64xbf16, #ttnn_layout3> {
    // CHECK: %[[DEVICE_OP:.*]] = "ttnn.get_device"
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.empty"(%0) <{dtype = #tt.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <<30x2>>, <interleaved>>, shape = #ttnn.shape<1x30x30x64>}> : (!ttnn.device) -> tensor<1x30x30x64xbf16, #ttnn_layout3>
    // CHECK: %[[TO_LAYOUT_INPUT:.*]] = "ttnn.to_layout"(%arg0, %[[DEVICE_OP]])
    // CHECK-SAME: dtype = #tt.supportedDataTypes<bf16>
    // CHECK-SAME: layout = #ttnn.layout<row_major>
    %3 = "ttnn.conv_transpose2d"(%arg0, %arg1, %arg2, %0) <{batch_size = 1 : i32, dilation = array<i32: 1, 1>, groups = 1 : i32, in_channels = 64 : i32, input_height = 32 : i32, input_width = 32 : i32, kernel_size = array<i32: 3, 3>, out_channels = 64 : i32, padding = array<i32: 0, 0>, stride = array<i32: 1, 1>, output_padding = array<i32: 0, 0>}> : (tensor<1x32x32x64xbf16, #ttnn_layout>, tensor<64x64x3x3xbf16, #ttnn_layout1>, tensor<1x1x1x64xbf16, #ttnn_layout2>, !ttnn.device) -> tensor<1x1x900x64xbf16, #ttnn_layout3>
    // CHECK-NEXT: %[[CONV2D_RESULT:.*]] = "ttnn.conv_transpose2d"(%[[TO_LAYOUT_INPUT]], %[[CONV2D_WEIGHTS:.*]], %[[CONV2D_BIAS:.*]], %[[DEVICE_OP]])
    // CHECK-NEXT: %[[TO_LAYOUT_OUTPUT:.*]] = "ttnn.to_layout"(%[[CONV2D_RESULT]], %[[DEVICE_OP]])
    // CHECK-SAME: dtype = #tt.supportedDataTypes<bf16>
    // CHECK-SAME: layout = #ttnn.layout<tile>
    %4 = "ttnn.reshape"(%3) <{shape = [1 : i32, 30 : i32, 30 : i32, 64 : i32]}> : (tensor<1x1x900x64xbf16, #ttnn_layout3>) -> tensor<1x30x30x64xbf16, #ttnn_layout3>
    return %4 : tensor<1x30x30x64xbf16, #ttnn_layout3>
  }

  func.func @conv2d_without_bias(%arg0: tensor<1x32x32x64xbf16, #ttnn_layout>, %arg1: tensor<64x64x3x3xbf16, #ttnn_layout1>) -> tensor<1x30x30x64xbf16, #ttnn_layout3> {
    // CHECK: %[[DEVICE_OP:.*]] = "ttnn.get_device"
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.empty"(%0) <{dtype = #tt.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <<30x2>>, <interleaved>>, shape = #ttnn.shape<1x30x30x64>}> : (!ttnn.device) -> tensor<1x30x30x64xbf16, #ttnn_layout3>
    %2 = "ttnn.reshape"(%arg0) <{shape = [1 : i32, 1 : i32, 1024 : i32, 64 : i32]}> : (tensor<1x32x32x64xbf16, #ttnn_layout>) -> tensor<1x1x1024x64xbf16, #ttnn_layout4>
    // CHECK: %[[TO_LAYOUT_INPUT:.*]] = "ttnn.to_layout"
    // CHECK-SAME: dtype = #tt.supportedDataTypes<bf16>
    // CHECK-SAME: layout = #ttnn.layout<row_major>
    %3 = "ttnn.conv2d"(%2, %arg1, %0) <{batch_size = 1 : i32, dilation = array<i32: 1, 1>, groups = 1 : i32, in_channels = 64 : i32, input_height = 32 : i32, input_width = 32 : i32, kernel_size = array<i32: 3, 3>, out_channels = 64 : i32, padding = array<i32: 0, 0>, stride = array<i32: 1, 1>}> : (tensor<1x1x1024x64xbf16, #ttnn_layout4>, tensor<64x64x3x3xbf16, #ttnn_layout1>, !ttnn.device) -> tensor<1x1x900x64xbf16, #ttnn_layout3>
    // CHECK-NEXT: %[[CONV2D_RESULT:.*]] = "ttnn.conv2d"(%[[TO_LAYOUT_INPUT]], %[[CONV2D_WEIGHTS:.*]], %[[DEVICE_OP]])
    // CHECK-NEXT: %[[TO_LAYOUT_OUTPUT:.*]] = "ttnn.to_layout"(%[[CONV2D_RESULT]], %[[DEVICE_OP]])
    // CHECK-SAME: dtype = #tt.supportedDataTypes<bf16>
    // CHECK-SAME: layout = #ttnn.layout<tile>
    %4 = "ttnn.reshape"(%3) <{shape = [1 : i32, 30 : i32, 30 : i32, 64 : i32]}> : (tensor<1x1x900x64xbf16, #ttnn_layout3>) -> tensor<1x30x30x64xbf16, #ttnn_layout3>
    return %4 : tensor<1x30x30x64xbf16, #ttnn_layout3>
  }

  func.func @conv_transpose2d_without_bias(%arg0: tensor<1x32x32x64xbf16, #ttnn_layout>, %arg1: tensor<64x64x3x3xbf16, #ttnn_layout1>) -> tensor<1x30x30x64xbf16, #ttnn_layout3> {
    // CHECK: %[[DEVICE_OP:.*]] = "ttnn.get_device"
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.empty"(%0) <{dtype = #tt.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <<30x2>>, <interleaved>>, shape = #ttnn.shape<1x30x30x64>}> : (!ttnn.device) -> tensor<1x30x30x64xbf16, #ttnn_layout3>
    // CHECK: %[[TO_LAYOUT_INPUT:.*]] = "ttnn.to_layout"(%arg0, %[[DEVICE_OP]])
    // CHECK-SAME: dtype = #tt.supportedDataTypes<bf16>
    // CHECK-SAME: layout = #ttnn.layout<row_major>
    %3 = "ttnn.conv_transpose2d"(%arg0, %arg1, %0) <{batch_size = 1 : i32, dilation = array<i32: 1, 1>, groups = 1 : i32, in_channels = 64 : i32, input_height = 32 : i32, input_width = 32 : i32, kernel_size = array<i32: 3, 3>, out_channels = 64 : i32, padding = array<i32: 0, 0>, stride = array<i32: 1, 1>, output_padding = array<i32: 0, 0>}> : (tensor<1x32x32x64xbf16, #ttnn_layout>, tensor<64x64x3x3xbf16, #ttnn_layout1>, !ttnn.device) -> tensor<1x1x900x64xbf16, #ttnn_layout3>
    // CHECK-NEXT: %[[CONV2D_RESULT:.*]] = "ttnn.conv_transpose2d"(%[[TO_LAYOUT_INPUT]], %[[CONV2D_WEIGHTS:.*]], %[[DEVICE_OP]])
    // CHECK-NEXT: %[[TO_LAYOUT_OUTPUT:.*]] = "ttnn.to_layout"(%[[CONV2D_RESULT]], %[[DEVICE_OP]])
    // CHECK-SAME: dtype = #tt.supportedDataTypes<bf16>
    // CHECK-SAME: layout = #ttnn.layout<tile>
    %4 = "ttnn.reshape"(%3) <{shape = [1 : i32, 30 : i32, 30 : i32, 64 : i32]}> : (tensor<1x1x900x64xbf16, #ttnn_layout3>) -> tensor<1x30x30x64xbf16, #ttnn_layout3>
    return %4 : tensor<1x30x30x64xbf16, #ttnn_layout3>
  }
}
