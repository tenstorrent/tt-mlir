// RUN: ttmlir-opt --tt-register-device="system-desc-path=%system_desc_path%" --ttnn-workaround --canonicalize %s | FileCheck %s

#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 + d1 + d2, d3), <1x1>, memref<1x2x!tt.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 12 + d1 + d2, d3), <1x1>, memref<1x2x!tt.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1x!tt.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 + d1 + d2, d3), <1x1>, memref<1x1x!tt.tile<32x32, f32>, #dram>, <interleaved>>
module @moreh_cumsum attributes {} {
  func.func @main(%arg0: tensor<1x1x1x46xbf16, #ttnn_layout>, %arg1: tensor<1x12x1x46xf32, #ttnn_layout1>, %arg2: tensor<1xf32, #ttnn_layout2>) -> tensor<1x12x1x46xf32, #ttnn_layout1> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.reshape"(%arg2) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1xf32, #ttnn_layout2>) -> tensor<1x1x1x1xf32, #ttnn_layout3>
    %2 = "ttnn.repeat"(%1) <{repeat_dims = #ttnn.shape<1x12x1x46>}> : (tensor<1x1x1x1xf32, #ttnn_layout3>) -> tensor<1x12x1x46xf32, #ttnn_layout1>
    // CHECK: %[[ARG0:[0-9]+]] = {{.*}}(%arg0,
    // CHECK-SAME: dtype = #tt.supportedDataTypes<f32>
    // CHECK-SAME: tensor<1x1x1x46xbf16
    // CHECK-SAME: -> tensor<1x1x1x46xf32
    // CHECK: "ttnn.where"(%[[ARG0]]
    // CHECK-SAME: tensor<1x1x1x46xf32
    // CHECK-SAME: tensor<1x12x1x46xf32
    // CHECK-SAME: tensor<1x12x1x46xf32
    // CHECK-SAME: -> tensor<1x12x1x46xf32
    %4 = "ttnn.where"(%arg0, %arg1, %2) : (tensor<1x1x1x46xbf16, #ttnn_layout>, tensor<1x12x1x46xf32, #ttnn_layout1>, tensor<1x12x1x46xf32, #ttnn_layout1>) -> tensor<1x12x1x46xf32, #ttnn_layout1>
    return %4 : tensor<1x12x1x46xf32, #ttnn_layout1>
  }
}
