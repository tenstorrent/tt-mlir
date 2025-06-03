//RUN: ttmlir-opt --tt-register-device --ttnn-merge-layouts-with-func-args --canonicalize --mlir-print-local-scope %s | FileCheck %s

#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout_rm_system = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x32xbf16, #system_memory>>
#ttnn_layout_rm_device = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x32xbf16, #dram>, <interleaved>>
#ttnn_layout_tile_device = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!tt.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout_tile_device_f32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!tt.tile<32x32, f32>, #dram>, <interleaved>>

module attributes {} {
  func.func @merge_to_layout_op_in_function_arg(%arg0: tensor<32x32xbf16, #ttnn_layout_tile_device>) -> tensor<32x32xbf16, #ttnn_layout_rm_system> {
    %1 = "ttnn.to_layout"(%arg0) <{layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#system_memory>}> : (tensor<32x32xbf16, #ttnn_layout_tile_device>) -> tensor<32x32xbf16, #ttnn_layout_rm_system>
    return %1 : tensor<32x32xbf16, #ttnn_layout_rm_system>
    // CHECK: return %arg0
    // CHECK-SAME: tensor<32x32xbf16
    // CHECK-SAME: #ttnn.buffer_type<system_memory>
  }

  func.func @merge_to_layout_ops_in_function_arg_multiple_same_uses(%arg0: tensor<32x32xbf16, #ttnn_layout_tile_device>) -> (tensor<32x32xbf16, #ttnn_layout_rm_system>, tensor<32x32xbf16, #ttnn_layout_rm_system>, tensor<32x32xbf16, #ttnn_layout_rm_system>) {
    %1 = "ttnn.to_layout"(%arg0) <{layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#system_memory>}> : (tensor<32x32xbf16, #ttnn_layout_tile_device>) -> tensor<32x32xbf16, #ttnn_layout_rm_system>
    %2 = "ttnn.to_layout"(%arg0) <{layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#system_memory>}> : (tensor<32x32xbf16, #ttnn_layout_tile_device>) -> tensor<32x32xbf16, #ttnn_layout_rm_system>
    %3 = "ttnn.to_layout"(%arg0) <{layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#system_memory>}> : (tensor<32x32xbf16, #ttnn_layout_tile_device>) -> tensor<32x32xbf16, #ttnn_layout_rm_system>
    return %1, %2, %3 : tensor<32x32xbf16, #ttnn_layout_rm_system>, tensor<32x32xbf16, #ttnn_layout_rm_system>, tensor<32x32xbf16, #ttnn_layout_rm_system>
    // CHECK: return %arg0, %arg0, %arg0
    // CHECK-SAME: tensor<32x32xbf16
    // CHECK-SAME: #ttnn.buffer_type<system_memory>
    // CHECK-SAME: tensor<32x32xbf16
    // CHECK-SAME: #ttnn.buffer_type<system_memory>
    // CHECK-SAME: tensor<32x32xbf16
    // CHECK-SAME: #ttnn.buffer_type<system_memory>
  }

  func.func @merge_to_layout_ops_in_function_arg_multiple_different_uses(%arg0: tensor<32x32xbf16, #ttnn_layout_tile_device>) -> (tensor<32x32xbf16, #ttnn_layout_rm_system>, tensor<32x32xbf16, #ttnn_layout_tile_device_f32>, tensor<32x32xbf16, #ttnn_layout_rm_system>) {
    %1 = "ttnn.to_layout"(%arg0) <{layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#system_memory>}> : (tensor<32x32xbf16, #ttnn_layout_tile_device>) -> tensor<32x32xbf16, #ttnn_layout_rm_system>
    %2 = "ttnn.to_layout"(%arg0) <{layout = #ttnn.layout<tile>, dtype = #tt.supportedDataTypes<f32>, memory_config = #ttnn.memory_config<#system_memory>}> : (tensor<32x32xbf16, #ttnn_layout_tile_device>) -> tensor<32x32xbf16, #ttnn_layout_tile_device_f32>
    %3 = "ttnn.to_layout"(%arg0) <{layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#system_memory>}> : (tensor<32x32xbf16, #ttnn_layout_tile_device>) -> tensor<32x32xbf16, #ttnn_layout_rm_system>
    return %1, %2, %3 : tensor<32x32xbf16, #ttnn_layout_rm_system>, tensor<32x32xbf16, #ttnn_layout_tile_device_f32>, tensor<32x32xbf16, #ttnn_layout_rm_system>
    // CHECK: "ttnn.to_layout"
    // CHECK-NEXT: "ttnn.to_layout"
    // CHECK-NEXT: "ttnn.to_layout"
    // CHECK-NEXT: return %0, %1, %2
  }

  // Verify that argument with no use isn't rewritten
  func.func @merge_to_layout_ops_in_function_arg_no_use(%arg0: tensor<32x32xbf16, #ttnn_layout_tile_device>, %arg1: tensor<32x32xbf16, #ttnn_layout_tile_device>) -> (tensor<32x32xbf16, #ttnn_layout_rm_system>) {
    %1 = "ttnn.to_layout"(%arg0) <{layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#system_memory>}> : (tensor<32x32xbf16, #ttnn_layout_tile_device>) -> tensor<32x32xbf16, #ttnn_layout_rm_system>
    return %1 : tensor<32x32xbf16, #ttnn_layout_rm_system>
    // CHECK: func.func @merge_to_layout_ops_in_function_arg_no_use
    // CHECK-SAME: %arg1: tensor<32x32xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!tt.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>
    // CHECK: return %arg0
  }
}
