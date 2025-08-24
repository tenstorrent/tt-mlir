// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttnn-workaround -o %t %s
// RUN: FileCheck %s --input-file=%t
#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 64 + d1, d2), <1x1>, memref<64x4x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 128 + d1, d2), <1x1>, memref<256x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
module attributes {} {
  func.func @permute_general(%arg0: tensor<32x64x128xsi32, #ttnn_layout>) -> tensor<64x128x32xsi32, #ttnn_layout1> {
    // CHECK: %[[LAYOUT_TYPE_CAST0:[0-9]+]] = "ttnn.to_layout"(%arg0
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<f32>
    // CHECK-SAME: -> tensor<32x64x128xf32
    // CHECK: %[[PERMUTE:[0-9]+]] = "ttnn.permute"(%[[LAYOUT_TYPE_CAST0]])
    %0 = "ttnn.permute"(%arg0) <{permutation = array<i64: 1, 2, 0>}> : (tensor<32x64x128xsi32, #ttnn_layout>) -> tensor<64x128x32xsi32, #ttnn_layout1>
    // CHECK: %[[LAYOUT_TYPE_CAST1:[0-9]+]] = "ttnn.to_layout"(%[[PERMUTE]]
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<si32>
    // CHECK-SAME: -> tensor<64x128x32xsi32
    return %0 : tensor<64x128x32xsi32, #ttnn_layout1>
  }
}
