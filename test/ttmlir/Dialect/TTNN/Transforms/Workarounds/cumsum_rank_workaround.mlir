// RUN: ttmlir-opt --ttcore-register-device --ttnn-workaround --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t
// UNSUPPORTED: true

#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x32xui32, #system_memory>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, u32>, #system_memory>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, u32>, #dram>, <interleaved>>
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x32xf32, #system_memory>>
#ttnn_layout4 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #system_memory>>
#ttnn_layout5 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout6 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 2048 + d1 * 64 + d2, d3), <1x1>, memref<2048x64xui32, #system_memory>>
#ttnn_layout7 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 2048 + d1 * 64 + d2, d3), <1x1>, memref<64x2x!ttcore.tile<32x32, u32>, #system_memory>>
#ttnn_layout8 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 2048 + d1 * 64 + d2, d3), <1x1>, memref<64x2x!ttcore.tile<32x32, u32>, #dram>, <interleaved>>
module @moreh_cumsum attributes {} {
  func.func public @test_cumsum_layout_reshape(%arg0: tensor<1x32xui32, #ttnn_layout>) -> tensor<1x32xui32, #ttnn_layout> {
    // CHECK-LABEL: func.func public @test_cumsum_layout_reshape
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.to_layout"(%arg0) <{layout = #ttnn.layout<tile>}> : (tensor<1x32xui32, #ttnn_layout>) -> tensor<1x32xui32, #ttnn_layout1>
    %2 = "ttnn.to_device"(%1, %0) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<1x32xui32, #ttnn_layout1>, !ttnn.device) -> tensor<1x32xui32, #ttnn_layout2>
    "ttnn.deallocate"(%1) <{force = false}> : (tensor<1x32xui32, #ttnn_layout1>) -> ()
    // CHECK: %[[RESHAPE:[0-9]+]] = "ttnn.reshape"
    // CHECK-SAME: {shape = [1 : i32, 32 : i32, 1 : i32, 1 : i32]}
    // CHECK-SAME: tensor<1x32xui32
    // CHECK-SAME: -> tensor<1x32x1x1xui32
    // CHECK: %[[CUMSUM:[0-9]+]] = "ttnn.moreh_cumsum"(%[[RESHAPE]])
    // CHECK-SAME: {dim = 0 : i64}
    // CHECK-SAME: tensor<1x32x1x1xui32
    // CHECK-SAME: -> tensor<1x32x1x1xui32
    %3 = "ttnn.moreh_cumsum"(%2) <{dim = 0 : i64}> : (tensor<1x32xui32, #ttnn_layout2>) -> tensor<1x32xui32, #ttnn_layout2>
    // CHECK: "ttnn.reshape"(%[[CUMSUM]])
    // CHECK-SAME: {shape = [1 : i32, 32 : i32]}
    // CHECK-SAME: tensor<1x32x1x1xui32
    // CHECK-SAME: -> tensor<1x32xui32
    "ttnn.deallocate"(%2) <{force = false}> : (tensor<1x32xui32, #ttnn_layout2>) -> ()
    %4 = "ttnn.from_device"(%3) : (tensor<1x32xui32, #ttnn_layout2>) -> tensor<1x32xui32, #ttnn_layout1>
    "ttnn.deallocate"(%3) <{force = false}> : (tensor<1x32xui32, #ttnn_layout2>) -> ()
    %5 = "ttnn.to_layout"(%4) <{layout = #ttnn.layout<row_major>}> : (tensor<1x32xui32, #ttnn_layout1>) -> tensor<1x32xui32, #ttnn_layout>
    "ttnn.deallocate"(%4) <{force = false}> : (tensor<1x32xui32, #ttnn_layout1>) -> ()
    return %5 : tensor<1x32xui32, #ttnn_layout>
  }

  func.func public @test_cumsum_reshape(%arg0: tensor<1x32xf32, #ttnn_layout3>) -> tensor<1x32xf32, #ttnn_layout3> {
    // CHECK-LABEL: func.func public @test_cumsum_reshape(
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.to_layout"(%arg0) <{layout = #ttnn.layout<tile>}> : (tensor<1x32xf32, #ttnn_layout3>) -> tensor<1x32xf32, #ttnn_layout4>
    %2 = "ttnn.to_device"(%1, %0) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<1x32xf32, #ttnn_layout4>, !ttnn.device) -> tensor<1x32xf32, #ttnn_layout5>
    "ttnn.deallocate"(%1) <{force = false}> : (tensor<1x32xf32, #ttnn_layout4>) -> ()
    // CHECK: %[[RESHAPE:[0-9]+]] = "ttnn.reshape"
    // CHECK-SAME: {shape = [1 : i32, 32 : i32, 1 : i32, 1 : i32]}
    // CHECK-SAME: tensor<1x32xf32
    // CHECK-SAME: -> tensor<1x32x1x1xf32
    // CHECK: %[[CUMSUM:[0-9]+]] = "ttnn.moreh_cumsum"(%[[RESHAPE]])
    // CHECK-SAME: {dim = 1 : i64}
    // CHECK-SAME: tensor<1x32x1x1xf32
    // CHECK-SAME: -> tensor<1x32x1x1xf32
    %3 = "ttnn.moreh_cumsum"(%2) <{dim = 1 : i64}> : (tensor<1x32xf32, #ttnn_layout5>) -> tensor<1x32xf32, #ttnn_layout5>
    // CHECK: "ttnn.reshape"(%[[CUMSUM]])
    // CHECK-SAME: {shape = [1 : i32, 32 : i32]}
    // CHECK-SAME: tensor<1x32x1x1xf32
    // CHECK-SAME: -> tensor<1x32xf32
    "ttnn.deallocate"(%2) <{force = false}> : (tensor<1x32xf32, #ttnn_layout5>) -> ()
    %4 = "ttnn.from_device"(%3) : (tensor<1x32xf32, #ttnn_layout5>) -> tensor<1x32xf32, #ttnn_layout4>
    "ttnn.deallocate"(%3) <{force = false}> : (tensor<1x32xf32, #ttnn_layout5>) -> ()
    %5 = "ttnn.to_layout"(%4) <{layout = #ttnn.layout<row_major>}> : (tensor<1x32xf32, #ttnn_layout4>) -> tensor<1x32xf32, #ttnn_layout3>
    "ttnn.deallocate"(%4) <{force = false}> : (tensor<1x32xf32, #ttnn_layout4>) -> ()
    return %5 : tensor<1x32xf32, #ttnn_layout3>
  }

  func.func public @test_cumsum_layout(%arg0: tensor<1x32x64x64xui32, #ttnn_layout6>) -> tensor<1x32x64x64xui32, #ttnn_layout6> {
    // CHECK-LABEL: func.func public @test_cumsum_layout(
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.to_layout"(%arg0) <{layout = #ttnn.layout<tile>}> : (tensor<1x32x64x64xui32, #ttnn_layout6>) -> tensor<1x32x64x64xui32, #ttnn_layout7>
    // CHECK: "ttnn.to_layout"
    %2 = "ttnn.to_device"(%1, %0) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<1x32x64x64xui32, #ttnn_layout7>, !ttnn.device) -> tensor<1x32x64x64xui32, #ttnn_layout8>
    "ttnn.deallocate"(%1) <{force = false}> : (tensor<1x32x64x64xui32, #ttnn_layout7>) -> ()
    // CHECK: %[[CUMSUM:[0-9]+]] = "ttnn.moreh_cumsum"
    // CHECK-SAME: <{dim = 1 : i64}>
    // CHECK-SAME: tensor<1x32x64x64xui32
    // CHECK-SAME: -> tensor<1x32x64x64xui32
    %3 = "ttnn.moreh_cumsum"(%2) <{dim = 1 : i64}> : (tensor<1x32x64x64xui32, #ttnn_layout8>) -> tensor<1x32x64x64xui32, #ttnn_layout8>
    "ttnn.deallocate"(%2) <{force = false}> : (tensor<1x32x64x64xui32, #ttnn_layout8>) -> ()
    %4 = "ttnn.from_device"(%3) : (tensor<1x32x64x64xui32, #ttnn_layout8>) -> tensor<1x32x64x64xui32, #ttnn_layout7>
    "ttnn.deallocate"(%3) <{force = false}> : (tensor<1x32x64x64xui32, #ttnn_layout8>) -> ()
    %5 = "ttnn.to_layout"(%4) <{layout = #ttnn.layout<row_major>}> : (tensor<1x32x64x64xui32, #ttnn_layout7>) -> tensor<1x32x64x64xui32, #ttnn_layout6>
    "ttnn.deallocate"(%4) <{force = false}> : (tensor<1x32x64x64xui32, #ttnn_layout7>) -> ()
    return %5 : tensor<1x32x64x64xui32, #ttnn_layout6>
  }
}
