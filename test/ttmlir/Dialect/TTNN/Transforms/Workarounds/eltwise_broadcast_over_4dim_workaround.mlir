// RUN: ttmlir-opt --split-input-file --tt-register-device --ttnn-workaround %s | FileCheck %s

#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1x!tt.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1, d2, d3, d4) -> (d0 * 640 + d1 * 160 + d2 * 32 + d3, d4), <1x1>, memref<60x1x!tt.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1, d2, d3, d4) -> (d0 * 32 + d1 * 32 + d2 * 32 + d3, d4), <1x1>, memref<1x1x!tt.tile<32x32, f32>, #dram>, <interleaved>>
module {
  tt.device_module {
    builtin.module attributes {} {
      tt.device @default_device = <workerGrid = #tt.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5] -> (0, 0, (((d0 * s1) * (s2 * s3) + d1 * (s2 * s3) + d2) floordiv s4) mod 12, ((d0 * s1) * (s2 * s3) + d1 * (s2 * s3) + d2) floordiv (s4 * 12) + ((d0 * s1) * (s2 * s3) + d1 * (s2 * s3) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
      func.func @simple_broadcast_over_4dim(%arg0: tensor<1xf32, #ttnn_layout>, %arg1: tensor<3x4x5x6x7xf32, #ttnn_layout1>) -> tensor<3x4x5x6x7xf32, #ttnn_layout1> {
        %0 = "ttnn.reshape"(%arg0) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1xf32, #ttnn_layout>) -> tensor<1x1x1x1x1xf32, #ttnn_layout2>
        // CHECK: "ttnn.reshape"
        // CHECK-NEXT: "ttnn.reshape"
        // CHECK-SAME: -> tensor<1x1x1x1xf32
        // CHECK: "ttnn.reshape"
        // CHECK-SAME: -> tensor<3x4x5x6x7
        %1 = "ttnn.repeat"(%0) <{repeat_dims = #ttnn.shape<3x4x5x6x7>}> : (tensor<1x1x1x1x1xf32, #ttnn_layout2>) -> tensor<3x4x5x6x7xf32, #ttnn_layout1>
        %2 = "ttnn.multiply"(%arg1, %1) : (tensor<3x4x5x6x7xf32, #ttnn_layout1>, tensor<3x4x5x6x7xf32, #ttnn_layout1>) -> tensor<3x4x5x6x7xf32, #ttnn_layout1>
        return %2 : tensor<3x4x5x6x7xf32, #ttnn_layout1>
      }
    }
  }
}

// -----

#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3, d4) -> (d0 * 160 + d1 * 160 + d2 * 32 + d3, d4), <1x1>, memref<15x1x!tt.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1, d2, d3, d4) -> (d0 * 640 + d1 * 160 + d2 * 32 + d3, d4), <1x1>, memref<60x1x!tt.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1, d2, d3, d4) -> (d0 * 640 + d1 * 160 + d2 * 32 + d3, d4), <1x1>, memref<60x1x!tt.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1, d2, d3, d4) -> (d0 * 640 + d1 * 160 + d2 * 32 + d3, d4), <1x1>, memref<60x1x!tt.tile<32x32, bf16>, #dram>, <interleaved>>
module attributes {} {
  tt.device @default_device = <workerGrid = #tt.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5] -> (0, 0, (((d0 * s1) * (s2 * s3) + d1 * (s2 * s3) + d2) floordiv s4) mod 12, ((d0 * s1) * (s2 * s3) + d1 * (s2 * s3) + d2) floordiv (s4 * 12) + ((d0 * s1) * (s2 * s3) + d1 * (s2 * s3) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
  func.func @implicit_broadcast_over_4dim(%arg0: tensor<3x1x5x1x7xf32, #ttnn_layout>, %arg1: tensor<3x4x5x6x7xf32, #ttnn_layout1>) -> tensor<3x4x5x6x7xbf16, #ttnn_layout3> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK-DAG: "ttnn.permute"{{.*}}-> tensor<4x6x3x5x7xf32
    // CHECK-DAG: "ttnn.permute"{{.*}}-> tensor<1x1x3x5x7xf32
    // CHECK-DAG: "ttnn.reshape"{{.*}}-> tensor<24x3x5x7xf32
    // CHECK-DAG: "ttnn.reshape"{{.*}}-> tensor<1x3x5x7xf32
    // CHECK: "ttnn.add"
    // CHECK-SAME: -> tensor<24x3x5x7xf32
    %1 = "ttnn.add"(%arg1, %arg0) : (tensor<3x4x5x6x7xf32, #ttnn_layout1>, tensor<3x1x5x1x7xf32, #ttnn_layout>) -> tensor<3x4x5x6x7xf32, #ttnn_layout2>
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: -> tensor<4x6x3x5x7xf32
    // CHECK: "ttnn.permute"
    // CHECK-SAME: -> tensor<3x4x5x6x7xf32
    %2 = "ttnn.to_layout"(%1, %0) <{dtype = #tt.supportedDataTypes<f32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<3x4x5x6x7xf32, #ttnn_layout2>, !ttnn.device) -> tensor<3x4x5x6x7xbf16, #ttnn_layout3>
    return %2 : tensor<3x4x5x6x7xbf16, #ttnn_layout3>
  }
}
