// RUN: ttmlir-opt --ttcore-register-device --ttnn-workaround --canonicalize %s | FileCheck %s
#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 2048 + d1 * 64 + d2, d3), <1x1>, memref<8192x3xf32, #system_memory>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 8192 + d1 * 128 + d2, d3), <1x1>, memref<32768x3xf32, #system_memory>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 2048 + d1 * 64 + d2, d3), <1x1>, memref<256x1x!ttcore.tile<32x32, f32>, #system_memory>>
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 2048 + d1 * 64 + d2, d3), <1x1>, memref<256x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout4 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 8192 + d1 * 128 + d2, d3), <1x1>, memref<1024x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout5 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 8192 + d1 * 128 + d2, d3), <1x1>, memref<1024x1x!ttcore.tile<32x32, f32>, #system_memory>>
module attributes {} {
  func.func @upsample2d_scale_unifrom(%arg0: tensor<4x32x64x3xf32, #ttnn_layout>) -> tensor<4x64x128x3xf32, #ttnn_layout1> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.to_layout"(%arg0) <{layout = #ttnn.layout<tile>}> : (tensor<4x32x64x3xf32, #ttnn_layout>) -> tensor<4x32x64x3xf32, #ttnn_layout2>
    %2 = "ttnn.to_device"(%1, %0) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<4x32x64x3xf32, #ttnn_layout2>, !ttnn.device) -> tensor<4x32x64x3xf32, #ttnn_layout3>
    "ttnn.deallocate"(%1) <{force = false}> : (tensor<4x32x64x3xf32, #ttnn_layout2>) -> ()
    // CHECK: "ttnn.to_layout"
    // CHECK: "ttnn.to_layout"
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK-SAME: layout = #ttnn.layout<row_major>
    // CHECK: "ttnn.upsample"
    %3 = "ttnn.upsample"(%2) <{mode = "nearest", scale_factor = 2 : si32}> : (tensor<4x32x64x3xf32, #ttnn_layout3>) -> tensor<4x64x128x3xf32, #ttnn_layout4>
    "ttnn.deallocate"(%2) <{force = false}> : (tensor<4x32x64x3xf32, #ttnn_layout3>) -> ()
    %4 = "ttnn.from_device"(%3) : (tensor<4x64x128x3xf32, #ttnn_layout4>) -> tensor<4x64x128x3xf32, #ttnn_layout5>
    "ttnn.deallocate"(%3) <{force = false}> : (tensor<4x64x128x3xf32, #ttnn_layout4>) -> ()
    %5 = "ttnn.to_layout"(%4) <{layout = #ttnn.layout<row_major>}> : (tensor<4x64x128x3xf32, #ttnn_layout5>) -> tensor<4x64x128x3xf32, #ttnn_layout1>
    "ttnn.deallocate"(%4) <{force = false}> : (tensor<4x64x128x3xf32, #ttnn_layout5>) -> ()
    return %5 : tensor<4x64x128x3xf32, #ttnn_layout1>
  }
}
