// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttnn-workaround --canonicalize %s | FileCheck %s
#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<16x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<32x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
module attributes {} {
  func.func @forward(%arg0: tensor<32x32xf32, #ttnn_layout>, %arg1: tensor<512x128xf32, #ttnn_layout1>) -> tensor<32x32x128xf32, #ttnn_layout2> {
    // CHECK: %[[DEVICE_OP:.*]] = "ttnn.get_device"
    // Check that the input operand is transformed into the row major layout.
    // CHECK-NEXT: %[[TO_LAYOUT_INPUT:.*]] = "ttnn.to_layout"
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<u32>
    // CHECK-SAME: layout = #ttnn.layout<row_major>
    // CHECK-SAME: memory_config = #ttnn.memory_config<#dram, <interleaved>>
    // CHECK-SAME: -> tensor<32x32xui32
    // Check that the data type of the weight operand is transformed in bf16.
    // CHECK-NEXT: %[[TO_LAYOUT_WEIGHTS:.*]] = "ttnn.to_layout"
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK-SAME: layout = #ttnn.layout<tile>
    // CHECK-SAME: memory_config = #ttnn.memory_config<#dram, <interleaved>>
    // CHECK-SAME: -> tensor<512x128xbf16
    %0 = "ttnn.embedding"(%arg0, %arg1) : (tensor<32x32xf32, #ttnn_layout>, tensor<512x128xf32, #ttnn_layout1>) -> tensor<32x32x128xf32, #ttnn_layout2>
    // CHECK-NEXT: %[[EMBEDDING_OP:.*]] = "ttnn.embedding"(%[[TO_LAYOUT_INPUT]], %[[TO_LAYOUT_WEIGHTS]])
    // Check that the output operand is transformed back into the f32 data type.
    // CHECK-NEXT: "ttnn.to_layout"(%[[EMBEDDING_OP]], %[[DEVICE_OP]])
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<f32>
    // CHECK-SAME: layout = #ttnn.layout<tile>
    // CHECK-SAME: memory_config = #ttnn.memory_config<#dram, <interleaved>>
    return %0 : tensor<32x32x128xf32, #ttnn_layout2>
  }
}
