// REQUIRES: opmodel
// RUN: ttmlir-opt --ttcore-register-device --ttnn-operation-validation-and-fallback %s -o %t.mlir
// RUN: FileCheck %s --input-file %t.mlir

#dram = #ttnn.buffer_type<dram>
#ttnn_layout_row_major_si32 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0, d1, d2, d3), <1x1>, memref<1x1xsi32, #dram>, <interleaved>>
#ttnn_layout_row_major_bf16 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0, d1, d2, d3), <1x1>, memref<32x32xbf16, #dram>, <interleaved>>
#ttnn_layout_tile_si32 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<32x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#ttnn_layout_tile_bf16 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<32x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module attributes {} {
  func.func @subtract_with_row_major_i32(%arg0: tensor<1x1x32x32xsi32, #ttnn_layout_row_major_si32>, %arg1: tensor<1x1x32x32xsi32, #ttnn_layout_row_major_si32>) -> tensor<1x1x32x32xbf16, #ttnn_layout_tile_bf16> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device

    // The op validation pass should detect that op works with int32 inputs, but the output layout
    // doesn't match the expected one, so it inserts a revert to tile layout.

    // CHECK: %[[SUB_RES:.*]] = "ttnn.subtract"
    // CHECK: "ttnn.to_layout"
    // CHECK-SAME: (%[[SUB_RES]]
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>

    %1 = "ttnn.subtract"(%arg0, %arg1) <{operandSegmentSizes = array<i32: 2, 0>, dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<1x1x32x32xsi32, #ttnn_layout_row_major_si32>, tensor<1x1x32x32xsi32, #ttnn_layout_row_major_si32>) -> tensor<1x1x32x32xbf16, #ttnn_layout_tile_bf16>

    return %1 : tensor<1x1x32x32xbf16, #ttnn_layout_tile_bf16>
  }

  func.func @add_with_row_major_inputs(%arg0: tensor<1x1x32x32xbf16, #ttnn_layout_row_major_bf16>, %arg1: tensor<1x1x32x32xbf16, #ttnn_layout_row_major_bf16>) -> tensor<1x1x32x32xbf16, #ttnn_layout_tile_bf16> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device

    // The op validation pass should detect that op works with bf16 inputs, but the output layout
    // doesn't match the expected one, i.e. op will return row-major layout. After the op, it
    // inserts a revert to tile layout.

    // CHECK: %[[ADD_RES:.*]] = "ttnn.add"
    // CHECK: "ttnn.to_layout"
    // CHECK-SAME: (%[[ADD_RES]]
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>

    %1 = "ttnn.add"(%arg0, %arg1) <{operandSegmentSizes = array<i32: 2, 0>, dtype = #ttcore.supportedDataTypes<bf16>}> : (tensor<1x1x32x32xbf16, #ttnn_layout_row_major_bf16>, tensor<1x1x32x32xbf16, #ttnn_layout_row_major_bf16>) -> tensor<1x1x32x32xbf16, #ttnn_layout_tile_bf16>

    return %1 : tensor<1x1x32x32xbf16, #ttnn_layout_tile_bf16>
  }
}
