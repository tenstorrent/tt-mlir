// REQUIRES: opmodel
// RUN: ttmlir-opt --ttcore-register-device --ttnn-optimizer="post-optimizer-validation-enabled=true" %s -o %t.mlir
// RUN: FileCheck %s --input-file %t.mlir

// Test that the post-optimizer validation analysis detects when binary ops
// can work with row-major inputs but produce different output layouts than expected,
// and automatically inserts output layout revert operations.

#dram = #ttnn.buffer_type<dram>
#ttnn_layout_row_major = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0, d1, d2, d3), <1x1>, memref<32x32xbf16, #dram>, <interleaved>>
#ttnn_layout_tile = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<32x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module attributes {} {
  func.func @add_with_row_major_inputs(%arg0: tensor<1x1x32x32xbf16, #ttnn_layout_row_major>, %arg1: tensor<1x1x32x32xbf16, #ttnn_layout_row_major>) -> tensor<1x1x32x32xbf16, #ttnn_layout_tile> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device

    // The post-optimizer validation determines that add can work with row-major inputs
    // but produces a different output layout than expected, so it inserts a revert operation
    
    // CHECK: "ttnn.add"
    // CHECK-NEXT: "ttnn.to_layout"
    // CHECK-SAME: layout = #ttnn.layout<tile>
    // CHECK-SAME: memory_config = #ttnn.memory_config<#dram, <interleaved>>

    %1 = "ttnn.add"(%arg0, %arg1) <{operandSegmentSizes = array<i32: 2, 0>, output_dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<1x1x32x32xbf16, #ttnn_layout_row_major>, tensor<1x1x32x32xbf16, #ttnn_layout_row_major>) -> tensor<1x1x32x32xbf16, #ttnn_layout_tile>

    return %1 : tensor<1x1x32x32xbf16, #ttnn_layout_tile>
  }
}