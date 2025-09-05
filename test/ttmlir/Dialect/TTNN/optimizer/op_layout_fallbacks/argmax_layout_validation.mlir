// REQUIRES: opmodel
// RUN: ttmlir-opt --ttcore-register-device --ttnn-operation-validation-and-fallback %s -o %t.mlir
// RUN: FileCheck %s --input-file %t.mlir
// UNSUPPORTED: true

// TODO: Enable this test once ArgMax is implemented in the TTNN OpModel lib.

// Test that the op validation analysis automatically detects
// that argmax op requires row-major layout and BFloat16 input, UInt32 output.

#dram = #ttnn.buffer_type<dram>
#ttnn_layout_tile_f32 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<32x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout_row_major_uint32 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0, d1, d2), <1x1>, memref<1x1xui32, #dram>, <interleaved>>

module attributes {} {
  func.func @argmax_with_tile_f32_input(%arg0: tensor<1x1x32x32xf32, #ttnn_layout_tile_f32>) -> tensor<1x1x32xui32, #ttnn_layout_row_major_uint32> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device

    // The op validation should detect that argmax requires:
    // 1. Row-major layout for input
    // 2. BFloat16 data type for input
    // 3. UInt32 data type for output

    // CHECK: "ttnn.to_layout"
    // CHECK-SAME: layout = #ttnn.layout<row_major>
    // CHECK-SAME: -> tensor<1x1x32x32xbf16,
    // CHECK-NEXT: "ttnn.argmax"

    %1 = "ttnn.argmax"(%arg0) <{dim = 3 : i32, keep_dim = false, use_multicore = false}> : (tensor<1x1x32x32xf32, #ttnn_layout_tile_f32>) -> tensor<1x1x32xui32, #ttnn_layout_row_major_uint32>

    return %1 : tensor<1x1x32xui32, #ttnn_layout_row_major_uint32>
  }
}
