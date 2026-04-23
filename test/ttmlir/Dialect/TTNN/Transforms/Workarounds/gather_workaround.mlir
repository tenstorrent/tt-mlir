// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttnn-workaround -o %t %s
// RUN: FileCheck %s --input-file=%t

#dram = #ttnn.buffer_type<dram>
// Row-major layouts for input and index (workaround should convert these to tile).
#ttnn_layout_input_rm  = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<5x3xf32, #dram>, <interleaved>>
#ttnn_layout_index_rm  = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x3xui32, #dram>, <interleaved>>
#ttnn_layout_output    = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
// Tiled layouts for input and index (workaround should be a no-op for these).
#ttnn_layout_input_tile = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout_index_tile = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, u32>, #dram>, <interleaved>>

module attributes {} {
  // Verify that the input and index tensors are converted to tiled layout when
  // they are in row-major layout.
  func.func @gather_row_major_inputs(
      %arg0: tensor<5x3xf32, #ttnn_layout_input_rm>,
      %arg1: tensor<2x3xui32, #ttnn_layout_index_rm>)
      -> tensor<2x3xf32, #ttnn_layout_output> {
    // CHECK-LABEL: func.func @gather_row_major_inputs
    // Check that the input operand is converted to tiled layout.
    // CHECK: %[[TO_LAYOUT_INPUT:.*]] = "ttnn.to_layout"(%arg0)
    // CHECK-SAME: layout = #ttnn.layout<tile>
    // CHECK-SAME: -> tensor<5x3xf32,
    // Check that the index operand is converted to tiled layout.
    // CHECK-NEXT: %[[TO_LAYOUT_INDEX:.*]] = "ttnn.to_layout"(%arg1)
    // CHECK-SAME: layout = #ttnn.layout<tile>
    // CHECK-SAME: -> tensor<2x3xui32,
    %0 = "ttnn.gather"(%arg0, %arg1)
        <{dim = 0 : i32}>
        : (tensor<5x3xf32, #ttnn_layout_input_rm>,
           tensor<2x3xui32, #ttnn_layout_index_rm>)
        -> tensor<2x3xf32, #ttnn_layout_output>
    // CHECK-NEXT: %[[GATHER:.*]] = "ttnn.gather"(%[[TO_LAYOUT_INPUT]], %[[TO_LAYOUT_INDEX]])
    return %0 : tensor<2x3xf32, #ttnn_layout_output>
  }

  // Verify that no to_layout ops are inserted when the input and index tensors
  // are already in tiled layout.
  func.func @gather_tiled_inputs(
      %arg0: tensor<5x3xf32, #ttnn_layout_input_tile>,
      %arg1: tensor<2x3xui32, #ttnn_layout_index_tile>)
      -> tensor<2x3xf32, #ttnn_layout_output> {
    // CHECK-LABEL: func.func @gather_tiled_inputs
    // CHECK-NOT: "ttnn.to_layout"
    // CHECK: "ttnn.gather"
    %0 = "ttnn.gather"(%arg0, %arg1)
        <{dim = 0 : i32}>
        : (tensor<5x3xf32, #ttnn_layout_input_tile>,
           tensor<2x3xui32, #ttnn_layout_index_tile>)
        -> tensor<2x3xf32, #ttnn_layout_output>
    return %0 : tensor<2x3xf32, #ttnn_layout_output>
  }
}
