// REQUIRES: opmodel
// RUN: ttmlir-opt --ttcore-register-device --ttnn-optimizer="post-optimizer-validation-enabled=true" %s -o %t.mlir
// RUN: FileCheck %s --input-file %t.mlir

// Test that the post-optimizer validation analysis automatically detects
// that binary ops require appropriate layout and data type workarounds.

#dram = #ttnn.buffer_type<dram>
#ttnn_layout_row_major_si32 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0, d1, d2, d3), <1x1>, memref<1x1xsi32, #dram>, <interleaved>>
#ttnn_layout_row_major_bf16 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0, d1, d2, d3), <1x1>, memref<32x32xbf16, #dram>, <interleaved>>
#ttnn_layout_tile_si32 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<32x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#ttnn_layout_tile_bf16 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<32x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module attributes {} {
  func.func @subtract_with_row_major_i32(%arg0: tensor<1x1x32x32xsi32, #ttnn_layout_row_major_si32>, %arg1: tensor<1x1x32x32xsi32, #ttnn_layout_row_major_si32>) -> tensor<1x1x32x32xbf16, #ttnn_layout_tile_bf16> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device

    // The post-optimizer validation should detect that subtract requires:
    // 1. Tile layout for proper operation
    // 2. BFloat16 data type for integer inputs

    // CHECK: "ttnn.to_layout"
    // CHECK: "ttnn.subtract"
    // CHECK: "ttnn.to_layout"

    %1 = "ttnn.subtract"(%arg0, %arg1) <{operandSegmentSizes = array<i32: 2, 0>, output_dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<1x1x32x32xsi32, #ttnn_layout_row_major_si32>, tensor<1x1x32x32xsi32, #ttnn_layout_row_major_si32>) -> tensor<1x1x32x32xbf16, #ttnn_layout_tile_bf16>

    return %1 : tensor<1x1x32x32xbf16, #ttnn_layout_tile_bf16>
  }

  func.func @multiply_different_shapes_i32(%arg0: tensor<1x1x32x32xsi32, #ttnn_layout_tile_si32>, %arg1: tensor<1x1x1x32xsi32, #ttnn_layout_tile_si32>) -> tensor<1x1x32x32xbf16, #ttnn_layout_tile_bf16> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device

    // The post-optimizer validation should detect that multiply with int32 and different shapes
    // requires BFloat16 for broadcasting to work correctly

    // CHECK: "ttnn.to_layout"
    // CHECK: "ttnn.multiply"

    %1 = "ttnn.multiply"(%arg0, %arg1) <{operandSegmentSizes = array<i32: 2, 0>, output_dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<1x1x32x32xsi32, #ttnn_layout_tile_si32>, tensor<1x1x1x32xsi32, #ttnn_layout_tile_si32>) -> tensor<1x1x32x32xbf16, #ttnn_layout_tile_bf16>

    return %1 : tensor<1x1x32x32xbf16, #ttnn_layout_tile_bf16>
  }

  func.func @add_with_row_major_inputs(%arg0: tensor<1x1x32x32xbf16, #ttnn_layout_row_major_bf16>, %arg1: tensor<1x1x32x32xbf16, #ttnn_layout_row_major_bf16>) -> tensor<1x1x32x32xbf16, #ttnn_layout_tile_bf16> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device

    // The post-optimizer validation determines that add can work with row-major inputs
    // but produces a different output layout than expected, so it inserts a revert operation
    
    // CHECK: "ttnn.add"
    // CHECK: "ttnn.to_layout"

    %1 = "ttnn.add"(%arg0, %arg1) <{operandSegmentSizes = array<i32: 2, 0>, output_dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<1x1x32x32xbf16, #ttnn_layout_row_major_bf16>, tensor<1x1x32x32xbf16, #ttnn_layout_row_major_bf16>) -> tensor<1x1x32x32xbf16, #ttnn_layout_tile_bf16>

    return %1 : tensor<1x1x32x32xbf16, #ttnn_layout_tile_bf16>
  }
}