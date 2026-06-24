// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttnn-workaround --mlir-print-local-scope -o %t %s
// RUN: FileCheck %s --input-file=%t

#dram = #ttnn.buffer_type<dram>
// Row-major layouts for input and index (workaround should convert these to tile).
#ttnn_layout_input_rm  = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<5x3xf32, #dram>, <interleaved>>
#ttnn_layout_index_rm  = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x3xui32, #dram>, <interleaved>>
#ttnn_layout_output    = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
// Tiled layouts for input and index (workaround should be a no-op for these).
#ttnn_layout_input_tile = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout_index_tile = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, u32>, #dram>, <interleaved>>
// Tiled int32 index (workaround should cast it to uint32).
#ttnn_layout_index_tile_i32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>

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
    // CHECK-SAME: -> tensor<5x3xf32,
    // CHECK-SAME: !ttcore.tile<32x32,
    // Check that the index operand is converted to tiled layout.
    // CHECK-NEXT: %[[TO_LAYOUT_INDEX:.*]] = "ttnn.to_layout"(%arg1)
    // CHECK-SAME: -> tensor<2x3xui32,
    // CHECK-SAME: !ttcore.tile<32x32,
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

  // Verify that an Int32 index tensor is wrapped in the fill-style protection
  // pattern: lanes whose original index was negative are replaced with NaN in
  // the output, and the inner gather sees a UInt32 (clamped) index.
  func.func @gather_int32_index(
      %arg0: tensor<5x3xf32, #ttnn_layout_input_tile>,
      %arg1: tensor<2x3xi32, #ttnn_layout_index_tile_i32>)
      -> tensor<2x3xf32, #ttnn_layout_output> {
    // CHECK-LABEL: func.func @gather_int32_index
    // CHECK: %[[DEVICE:.*]] = "ttnn.get_device"
    // %zero = ttnn.full(0 : si32)
    // CHECK: %[[ZERO:.*]] = "ttnn.full"(%[[DEVICE]])
    // CHECK-SAME: fill_value = 0 : i32
    // %mask = ttnn.lt(idx, zero)  -> numeric predicate tensor
    // CHECK: %[[MASK:.*]] = "ttnn.lt"(%arg1, %[[ZERO]])
    // %safe = ttnn.maximum(idx, zero)  -> si32 (negatives clamped to 0)
    // CHECK: %[[CLAMPED:.*]] = "ttnn.maximum"(%arg1, %[[ZERO]])
    // %safe_u32 = ttnn.to_layout(safe) -> ui32
    // CHECK: %[[SAFE_U32:.*]] = "ttnn.to_layout"(%[[CLAMPED]])
    // CHECK-SAME: -> tensor<2x3xui32,
    // %raw = ttnn.gather(input, safe_u32, dim)
    // CHECK: %[[RAW:.*]] = "ttnn.gather"(%arg0, %[[SAFE_U32]])
    // %nan = ttnn.full(NaN : f32)   ; 0x7FC00000 is the bit pattern for quiet NaN
    // CHECK: %[[NAN:.*]] = "ttnn.full"(%[[DEVICE]])
    // CHECK-SAME: fill_value = 0x7FC00000 : f32
    // %result = ttnn.where(mask, NaN, raw)
    // CHECK: "ttnn.where"({{.*}}, %[[NAN]], %[[RAW]])
    %0 = "ttnn.gather"(%arg0, %arg1)
        <{dim = 0 : i32}>
        : (tensor<5x3xf32, #ttnn_layout_input_tile>,
           tensor<2x3xi32, #ttnn_layout_index_tile_i32>)
        -> tensor<2x3xf32, #ttnn_layout_output>
    return %0 : tensor<2x3xf32, #ttnn_layout_output>
  }
}
