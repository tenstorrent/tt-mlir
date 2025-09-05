// REQUIRES: opmodel
// RUN: ttmlir-opt --ttcore-register-device --ttnn-operation-validation-and-fallback %s -o %t.mlir
// RUN: FileCheck %s --input-file %t.mlir

#dram = #ttnn.buffer_type<dram>
#ttnn_layout_tile_si32 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<32x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#ttnn_layout_tile_bf16 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<1x1xbf16, #dram>, <interleaved>>

module attributes {} {
  func.func @sum_with_integer_input(%arg0: tensor<1x1x32x32xsi32, #ttnn_layout_tile_si32>) -> tensor<1x1x1x1xbf16, #ttnn_layout_tile_bf16> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device

    // The op validation pass should detect that sum will return si32 output,
    // but the output layout doesn't match the expected one, so it inserts a revert to row-major layout with bf16.

    // CHECK: %[[SUM_RES:.*]] = "ttnn.sum"
    // CHECK: "ttnn.to_layout"
    // CHECK-SAME: (%[[SUM_RES]]
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK-SAME: layout = #ttnn.layout<row_major>

    %1 = "ttnn.sum"(%arg0) <{
      dim_list = [2 : i32, 3 : i32],
      keep_dim = true
    }> : (tensor<1x1x32x32xsi32, #ttnn_layout_tile_si32>) -> tensor<1x1x1x1xbf16, #ttnn_layout_tile_bf16>

    return %1 : tensor<1x1x1x1xbf16, #ttnn_layout_tile_bf16>
  }
}
