// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttnn-workaround  %s -o slice_workaround_ttnn.mlir
// RUN: FileCheck %s --input-file=slice_workaround_ttnn.mlir

#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<4x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<2x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
module attributes {} {
  func.func @test_strided_slice_workaround(%arg0: tensor<4x32x32xf32, #ttnn_layout>) -> tensor<2x16x8xf32, #ttnn_layout1> {
    // CHECK-LABEL: @test_strided_slice_workaround(
    // CHECK: %[[ARG0:[0-9]+]] = "ttnn.to_layout"(%arg0)
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK-SAME: tensor<4x32x32xf32
    // CHECK-SAME: -> tensor<4x32x32xbf16
    // CHECK: %[[SLICE:[0-9]+]] = "ttnn.slice"(%[[ARG0]])
    // CHECK-SAME: begins = [0 : i32, 0 : i32, 0 : i32]
    // CHECK-SAME: ends = [2 : i32, 16 : i32, 16 : i32]
    // CHECK-SAME: step = [1 : i32, 1 : i32, 2 : i32]}
    // CHECK-SAME: tensor<4x32x32xbf16
    // CHECK-SAME: -> tensor<2x16x8xbf16
    %0 = "ttnn.slice"(%arg0) <{begins = [0 : i32, 0 : i32, 0 : i32], ends = [2 : i32, 16 : i32, 16 : i32], step = [1 : i32, 1 : i32, 2 : i32]}> : (tensor<4x32x32xf32, #ttnn_layout>) -> tensor<2x16x8xf32, #ttnn_layout1>
    // CHECK: "ttnn.to_layout"(%[[SLICE]])
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<f32>
    // CHECK-SAME: tensor<2x16x8xbf16
    // CHECK-SAME: -> tensor<2x16x8xf32
    return %0 : tensor<2x16x8xf32, #ttnn_layout1>
  }
}
