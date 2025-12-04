// RUN: ttmlir-opt --ttnn-convert-to-golden %s | FileCheck %s

#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout_host = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x128xbf16, #system_memory>>

// Test basic conversion: single input, single output, simple add op
// Verify from_torch has layout, dtype, and memory_config attributes
// CHECK-LABEL: @simple_add
func.func @simple_add(%arg0: tensor<64x128xbf16, #ttnn_layout>) -> tensor<64x128xbf16, #ttnn_layout> {
    // CHECK: %[[TO_TORCH:.*]] = "ttnn.to_torch"(%arg0)
    // CHECK: %[[ADD:.*]] = "ttnn.add"(%[[TO_TORCH]], %[[TO_TORCH]])
    // CHECK-SAME: use_golden
    // CHECK: %[[FROM_TORCH:.*]] = "ttnn.from_torch"(%[[ADD]])
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK-SAME: layout = #ttnn.layout<tile>
    // CHECK-SAME: memory_config = #ttnn.memory_config<#dram, <interleaved>>
    // CHECK-SAME: -> tensor<64x128xbf16,
    // CHECK: return %[[FROM_TORCH]]
    %0 = "ttnn.add"(%arg0, %arg0) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<64x128xbf16, #ttnn_layout>, tensor<64x128xbf16, #ttnn_layout>) -> tensor<64x128xbf16, #ttnn_layout>
    return %0 : tensor<64x128xbf16, #ttnn_layout>
}

// Test multiple inputs
// CHECK-LABEL: @multiple_inputs
func.func @multiple_inputs(%arg0: tensor<64x128xbf16, #ttnn_layout>, %arg1: tensor<64x128xbf16, #ttnn_layout>) -> tensor<64x128xbf16, #ttnn_layout> {
    // CHECK: %[[TO_TORCH_0:.*]] = "ttnn.to_torch"(%arg0)
    // CHECK: %[[TO_TORCH_1:.*]] = "ttnn.to_torch"(%arg1)
    // CHECK: %[[MUL:.*]] = "ttnn.multiply"(%[[TO_TORCH_0]], %[[TO_TORCH_1]])
    // CHECK-SAME: use_golden
    // CHECK: %[[FROM_TORCH:.*]] = "ttnn.from_torch"(%[[MUL]])
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK-SAME: layout = #ttnn.layout<tile>
    // CHECK-SAME: memory_config = #ttnn.memory_config<#dram, <interleaved>>
    // CHECK: return %[[FROM_TORCH]]
    %0 = "ttnn.multiply"(%arg0, %arg1) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<64x128xbf16, #ttnn_layout>, tensor<64x128xbf16, #ttnn_layout>) -> tensor<64x128xbf16, #ttnn_layout>
    return %0 : tensor<64x128xbf16, #ttnn_layout>
}

// Test chained operations
// CHECK-LABEL: @chained_ops
func.func @chained_ops(%arg0: tensor<64x128xbf16, #ttnn_layout>, %arg1: tensor<64x128xbf16, #ttnn_layout>) -> tensor<64x128xbf16, #ttnn_layout> {
    // CHECK: %[[TO_TORCH_0:.*]] = "ttnn.to_torch"(%arg0)
    // CHECK: %[[TO_TORCH_1:.*]] = "ttnn.to_torch"(%arg1)
    // CHECK: %[[ADD:.*]] = "ttnn.add"(%[[TO_TORCH_0]], %[[TO_TORCH_1]])
    // CHECK-SAME: use_golden
    // CHECK: %[[RELU:.*]] = "ttnn.relu"(%[[ADD]])
    // CHECK-SAME: use_golden
    // CHECK: %[[FROM_TORCH:.*]] = "ttnn.from_torch"(%[[RELU]])
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK-SAME: layout = #ttnn.layout<tile>
    // CHECK-SAME: memory_config = #ttnn.memory_config<#dram, <interleaved>>
    // CHECK: return %[[FROM_TORCH]]
    %0 = "ttnn.add"(%arg0, %arg1) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<64x128xbf16, #ttnn_layout>, tensor<64x128xbf16, #ttnn_layout>) -> tensor<64x128xbf16, #ttnn_layout>
    %1 = "ttnn.relu"(%0) : (tensor<64x128xbf16, #ttnn_layout>) -> tensor<64x128xbf16, #ttnn_layout>
    return %1 : tensor<64x128xbf16, #ttnn_layout>
}

// Test multiple outputs - both should have correct layout attributes
// CHECK-LABEL: @multiple_outputs
func.func @multiple_outputs(%arg0: tensor<64x128xbf16, #ttnn_layout>, %arg1: tensor<64x128xbf16, #ttnn_layout>) -> (tensor<64x128xbf16, #ttnn_layout>, tensor<64x128xbf16, #ttnn_layout>) {
    // CHECK: %[[TO_TORCH_0:.*]] = "ttnn.to_torch"(%arg0)
    // CHECK: %[[TO_TORCH_1:.*]] = "ttnn.to_torch"(%arg1)
    // CHECK: %[[ADD:.*]] = "ttnn.add"(%[[TO_TORCH_0]], %[[TO_TORCH_1]])
    // CHECK-SAME: use_golden
    // CHECK: %[[MUL:.*]] = "ttnn.multiply"(%[[TO_TORCH_0]], %[[TO_TORCH_1]])
    // CHECK-SAME: use_golden
    // CHECK: %[[FROM_TORCH_0:.*]] = "ttnn.from_torch"(%[[ADD]])
    // CHECK-SAME: layout = #ttnn.layout<tile>
    // CHECK-SAME: memory_config = #ttnn.memory_config<#dram, <interleaved>>
    // CHECK: %[[FROM_TORCH_1:.*]] = "ttnn.from_torch"(%[[MUL]])
    // CHECK-SAME: layout = #ttnn.layout<tile>
    // CHECK-SAME: memory_config = #ttnn.memory_config<#dram, <interleaved>>
    // CHECK: return %[[FROM_TORCH_0]], %[[FROM_TORCH_1]]
    %0 = "ttnn.add"(%arg0, %arg1) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<64x128xbf16, #ttnn_layout>, tensor<64x128xbf16, #ttnn_layout>) -> tensor<64x128xbf16, #ttnn_layout>
    %1 = "ttnn.multiply"(%arg0, %arg1) <{dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<64x128xbf16, #ttnn_layout>, tensor<64x128xbf16, #ttnn_layout>) -> tensor<64x128xbf16, #ttnn_layout>
    return %0, %1 : tensor<64x128xbf16, #ttnn_layout>, tensor<64x128xbf16, #ttnn_layout>
}

// Test unary operation
// CHECK-LABEL: @unary_op
func.func @unary_op(%arg0: tensor<64x128xbf16, #ttnn_layout>) -> tensor<64x128xbf16, #ttnn_layout> {
    // CHECK: %[[TO_TORCH:.*]] = "ttnn.to_torch"(%arg0)
    // CHECK: %[[NEG:.*]] = "ttnn.neg"(%[[TO_TORCH]])
    // CHECK-SAME: use_golden
    // CHECK: %[[FROM_TORCH:.*]] = "ttnn.from_torch"(%[[NEG]])
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK-SAME: layout = #ttnn.layout<tile>
    // CHECK-SAME: memory_config = #ttnn.memory_config<#dram, <interleaved>>
    // CHECK: return %[[FROM_TORCH]]
    %0 = "ttnn.neg"(%arg0) : (tensor<64x128xbf16, #ttnn_layout>) -> tensor<64x128xbf16, #ttnn_layout>
    return %0 : tensor<64x128xbf16, #ttnn_layout>
}

// Test with row-major layout on host
// CHECK-LABEL: @row_major_host
func.func @row_major_host(%arg0: tensor<64x128xbf16, #ttnn_layout_host>) -> tensor<64x128xbf16, #ttnn_layout_host> {
    // CHECK: %[[TO_TORCH:.*]] = "ttnn.to_torch"(%arg0)
    // CHECK: %[[NEG:.*]] = "ttnn.neg"(%[[TO_TORCH]])
    // CHECK-SAME: use_golden
    // CHECK: %[[FROM_TORCH:.*]] = "ttnn.from_torch"(%[[NEG]])
    // CHECK-SAME: layout = #ttnn.layout<row_major>
    // CHECK-SAME: memory_config = #ttnn.memory_config<#system_memory>
    // CHECK: return %[[FROM_TORCH]]
    %0 = "ttnn.neg"(%arg0) : (tensor<64x128xbf16, #ttnn_layout_host>) -> tensor<64x128xbf16, #ttnn_layout_host>
    return %0 : tensor<64x128xbf16, #ttnn_layout_host>
}
