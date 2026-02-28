// REQUIRES: opmodel
// RUN: ttmlir-opt --ttcore-register-device --ttcore-mark-functions-as-forward --ttnn-operation-validation-and-fallback %s -o %t.mlir
// RUN: FileCheck %s --input-file %t.mlir

// Test that the op validation pass correctly handles sort op which produces
// multiple outputs (values and indices) with different layouts. The backend
// returns ui16 for indices, but the expected output layout is si32 tile.
// The pass should insert a to_layout revert for the indices output.

#dram = #ttnn.buffer_type<dram>
#ttnn_layout_tile_bf16 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout_tile_si32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>

module attributes {} {
  func.func @sort_multiple_outputs_layout_mismatch(%arg0: tensor<128x32xbf16, #ttnn_layout_tile_bf16>) -> (tensor<128x32xbf16, #ttnn_layout_tile_bf16>, tensor<128x32xsi32, #ttnn_layout_tile_si32>) {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device

    // CHECK: %[[VALUES:.*]], %[[INDICES:.*]] = "ttnn.sort"
    // CHECK: "ttnn.to_layout"
    // CHECK-SAME: (%[[INDICES]]
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<si32>

    %values, %indices = "ttnn.sort"(%arg0) <{descending = true, dim = 1 : si8, stable = false}> : (tensor<128x32xbf16, #ttnn_layout_tile_bf16>) -> (tensor<128x32xbf16, #ttnn_layout_tile_bf16>, tensor<128x32xsi32, #ttnn_layout_tile_si32>)

    return %values, %indices : tensor<128x32xbf16, #ttnn_layout_tile_bf16>, tensor<128x32xsi32, #ttnn_layout_tile_si32>
  }
}
