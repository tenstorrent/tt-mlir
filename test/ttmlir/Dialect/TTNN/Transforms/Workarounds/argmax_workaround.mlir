// RUN: ttmlir-opt --ttcore-register-device --ttnn-workaround --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x1x!ttcore.tile<32x32, u32>, #dram>, <interleaved>>
module attributes {} {
  func.func public @argmax_2d(%arg0: tensor<64x64xf32, #ttnn_layout>) -> tensor<64x1xui32, #ttnn_layout1> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: %[[PRE_RESHAPE:[0-9]+]] = "ttnn.reshape"(%arg0)
    // CHECK-SAME: {shape = [1 : i32, 1 : i32, 64 : i32, 64 : i32]}
    // CHECK-SAME: tensor<64x64xf32,
    // CHECK-SAME: -> tensor<1x1x64x64xf32
    // CHECK: %[[ARG0:[0-9]+]] = "ttnn.to_layout"(%[[PRE_RESHAPE]])
    // CHECK-SAME: layout = #ttnn.layout<row_major>
    // CHECK-SAME: tensor<1x1x64x64xf32,
    // CHECK-SAME: -> tensor<1x1x64x64xf32,
    %1 = "ttnn.empty"(%0) <{dtype = #ttcore.supportedDataTypes<u32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>, shape = #ttnn.shape<64x1>}> : (!ttnn.device) -> tensor<64x1xui32, #ttnn_layout1>
    // CHECK: [[ARG_MAX:[0-9]+]] = "ttnn.argmax"(%[[ARG0]])
    // CHECK-SAME: {dim = 3 : i32, keep_dim = false, use_multicore = false}
    // CHECK-SAME: tensor<1x1x64x64xf32
    // CHECK-SAME: -> tensor<1x1x64x1xui32
    %2 = "ttnn.argmax"(%arg0) <{dim = 1 : i32, use_multicore = false}> : (tensor<64x64xf32, #ttnn_layout>) -> tensor<64x1xui32, #ttnn_layout1>
    // CHECK: %[[TO_LAYOUT:[0-9]+]] = "ttnn.to_layout"(%[[ARG_MAX]])
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<u32>
    // CHECK-SAME: layout = #ttnn.layout<tile>
    // CHECK-SAME: (tensor<1x1x64x1xui32
    // CHECK-SAME: -> tensor<1x1x64x1xui32
    // CHECK: = "ttnn.reshape"(%[[TO_LAYOUT]])
    // CHECK-SAME: {shape = [64 : i32, 1 : i32]}
    // CHECK-SAME: tensor<1x1x64x1xui32
    // CHECK-SAME: -> tensor<64x1xui32
    return %2 : tensor<64x1xui32, #ttnn_layout1>
  }
}
