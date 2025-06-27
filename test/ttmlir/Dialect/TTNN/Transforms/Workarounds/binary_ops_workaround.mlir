// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttnn-workaround --canonicalize %s | FileCheck %s

#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x2x!ttcore.tile<32x32, u16>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x2x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x4xbf16, #dram>, <interleaved>>
module @jit_transpose attributes {} {
  func.func public @test_add_workaround(%arg0: tensor<32x64xui16, #ttnn_layout>, %arg1: tensor<32x64xui16, #ttnn_layout>) -> tensor<32x64xui16, #ttnn_layout> {
    // CHECK-LABEL: func.func public @test_add_workaround
    // CHECK: %[[ARG0:[0-9]+]] = "ttnn.to_layout"(%arg0)
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>,
    // CHECK-SAME: tensor<32x64xui16,
    // CHECK-SAME: -> tensor<32x64xbf16,
    // CHECK: %[[ARG1:[0-9]+]] = "ttnn.to_layout"(%arg1)
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>,
    // CHECK-SAME: tensor<32x64xui16,
    // CHECK-SAME: -> tensor<32x64xbf16,
    // CHECK: %[[ADD:[0-9]+]] = "ttnn.add"(%[[ARG0]], %[[ARG1]])
    // CHECK-SAME: tensor<32x64xbf16,
    // CHECK-SAME: tensor<32x64xbf16,
    // CHECK-SAME: -> tensor<32x64xbf16,
    %0 = "ttnn.add"(%arg0, %arg1) : (tensor<32x64xui16, #ttnn_layout>, tensor<32x64xui16, #ttnn_layout>) -> tensor<32x64xui16, #ttnn_layout>
    // CHECK: = "ttnn.to_layout"(%[[ADD]])
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<u16>,
    // CHECK-SAME:tensor<32x64xbf16,
    // CHECK-SAME: -> tensor<32x64xui16,
    return %0 : tensor<32x64xui16, #ttnn_layout>
  }

  func.func public @test_multiply_workaround(%arg0: tensor<32x64xsi32, #ttnn_layout1>, %arg1: tensor<32x64xsi32, #ttnn_layout1>) -> tensor<32x64xsi32, #ttnn_layout1> {
    // CHECK-LABEL: func.func public @test_multiply_workaround
    // CHECK: %[[ARG0:[0-9]+]] = "ttnn.to_layout"(%arg0)
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>,
    // CHECK-SAME: tensor<32x64xsi32,
    // CHECK-SAME: -> tensor<32x64xbf16,
    // CHECK: %[[ARG1:[0-9]+]] = "ttnn.to_layout"(%arg1)
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>,
    // CHECK-SAME: tensor<32x64xsi32,
    // CHECK-SAME: -> tensor<32x64xbf16,
    // CHECK: %[[MULTIPLY:[0-9]+]] = "ttnn.multiply"(%[[ARG0]], %[[ARG1]])
    // CHECK-SAME: tensor<32x64xbf16,
    // CHECK-SAME: tensor<32x64xbf16,
    // CHECK-SAME: -> tensor<32x64xbf16,
    %0 = "ttnn.multiply"(%arg0, %arg1) : (tensor<32x64xsi32, #ttnn_layout1>, tensor<32x64xsi32, #ttnn_layout1>) -> tensor<32x64xsi32, #ttnn_layout1>
    // CHECK: = "ttnn.to_layout"(%[[MULTIPLY]]
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<si32>,
    // CHECK-SAME:tensor<32x64xbf16,
    // CHECK-SAME: -> tensor<32x64xsi32,
    return %0 : tensor<32x64xsi32, #ttnn_layout1>
  }

  func.func public @test_bitwise_and_workaround(%arg0: tensor<32x64xf32, #ttnn_layout2>, %arg1: tensor<32x64xf32, #ttnn_layout2>) -> tensor<32x64xf32, #ttnn_layout2> {
    // CHECK-LABEL: func.func public @test_bitwise_and_workaround
    // CHECK: %[[ARG0:[0-9]+]] = "ttnn.to_layout"(%arg0)
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<si32>,
    // CHECK-SAME: tensor<32x64xf32,
    // CHECK-SAME: -> tensor<32x64xsi32
    // CHECK: %[[ARG1:[0-9]+]] = "ttnn.to_layout"(%arg1)
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<si32>,
    // CHECK-SAME: tensor<32x64xf32,
    // CHECK-SAME: -> tensor<32x64xsi32
    // CHECK: %[[AND:[0-9]+]] = "ttnn.bitwise_and"(%[[ARG0]], %[[ARG1]])
    // CHECK-SAME: tensor<32x64xsi32,
    // CHECK-SAME: tensor<32x64xsi32,
    // CHECK-SAME: -> tensor<32x64xsi32,
    %0 = "ttnn.bitwise_and"(%arg0, %arg1) : (tensor<32x64xf32, #ttnn_layout2>, tensor<32x64xf32, #ttnn_layout2>) -> tensor<32x64xf32, #ttnn_layout2>
    // CHECK: = "ttnn.to_layout"(%[[AND]])
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<f32>,
    // CHECK-SAME:tensor<32x64xsi32,
    // CHECK-SAME: -> tensor<32x64xf32,
    return %0 : tensor<32x64xf32, #ttnn_layout2>
  }

  func.func public @test_layout_workaround(%arg0: tensor<4x4xbf16, #ttnn_layout3>, %arg1: tensor<4x4xbf16, #ttnn_layout3>) -> tensor<4x4xbf16, #ttnn_layout3> {
    // CHECK-LABEL: func.func public @test_layout_workaround
    // CHECK: %[[ARG0:[0-9]+]] = "ttnn.to_layout"(%arg0)
    // CHECK-SAME: layout = #ttnn.layout<tile>,
    // CHECK-SAME: tensor<4x4xbf16,
    // CHECK-SAME: -> tensor<4x4xbf16,
    // CHECK: %[[ARG1:[0-9]+]] = "ttnn.to_layout"(%arg1)
    // CHECK-SAME: layout = #ttnn.layout<tile>,
    // CHECK-SAME: tensor<4x4xbf16,
    // CHECK-SAME: -> tensor<4x4xbf16,
    // CHECK: %[[ADD:[0-9]+]] = "ttnn.add"(%[[ARG0]], %[[ARG1]])
    // CHECK-SAME: tensor<4x4xbf16,
    // CHECK-SAME: tensor<4x4xbf16,
    // CHECK-SAME: -> tensor<4x4xbf16,
    %0 = "ttnn.add"(%arg0, %arg1) : (tensor<4x4xbf16, #ttnn_layout3>, tensor<4x4xbf16, #ttnn_layout3>) -> tensor<4x4xbf16, #ttnn_layout3>
    // CHECK: = "ttnn.to_layout"(%[[ADD]])
    // CHECK-SAME: layout = #ttnn.layout<row_major>,
    // CHECK-SAME: tensor<4x4xbf16,
    // CHECK-SAME: -> tensor<4x4xbf16,
    return %0 : tensor<4x4xbf16, #ttnn_layout3>
  }
}
