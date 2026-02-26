// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttnn-workaround -o %t %s
// RUN: FileCheck %s --input-file=%t

#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 30 + d1, d2), <1x1>, memref<1x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<1x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
module attributes {} {
  func.func @test_pad_workaround(%arg0: tensor<1x30x30xsi32, #ttnn_layout>) -> tensor<1x32x32xsi32, #ttnn_layout1> {
    // CHECK-LABEL: @test_pad_workaround
    // CHECK: %[[ARG0:[0-9]+]] = "ttnn.to_layout"(%arg0)
    // CHECK-SAME: <{dtype = #ttcore.supportedDataTypes<si32>,
    // CHECK-SAME: layout = #ttnn.layout<row_major>,
    // CHECK-SAME: tensor<1x30x30xsi32,
    // CHECK-SAME: -> tensor<1x30x30xsi32,
    // CHECK: %[[PAD:[0-9]+]] = "ttnn.pad"(%[[ARG0]])
    // CHECK-SAME: tensor<1x30x30xsi32
    // CHECK-SAME: -> tensor<1x32x32xsi32,
    %0 = "ttnn.pad"(%arg0) <{memory_config = #ttnn.memory_config<#dram, <interleaved>>, padding = array<i32: 0, 0, 1, 1, 1, 1>, use_multicore = true, value = 0.000000e+00 : f32}> : (tensor<1x30x30xsi32, #ttnn_layout>) -> tensor<1x32x32xsi32, #ttnn_layout1>
    // CHECK: %{{[0-9]+}} = "ttnn.to_layout"(%[[PAD]])
    // CHECK-SAME: <{dtype = #ttcore.supportedDataTypes<si32>,
    // CHECK-SAME: layout = #ttnn.layout<tile>
    // CHECK-SAME: tensor<1x32x32xsi32,
    // CHECK-SAME: -> tensor<1x32x32xsi32,
    return %0 : tensor<1x32x32xsi32, #ttnn_layout1>
  }
}
