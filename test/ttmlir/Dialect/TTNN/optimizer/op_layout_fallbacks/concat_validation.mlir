// REQUIRES: opmodel
// RUN: ttmlir-opt --ttcore-register-device --ttnn-operation-validation-and-fallback %s -o %t.mlir
// RUN: FileCheck %s --input-file %t.mlir

#dram = #ttnn.buffer_type<dram>
#ttnn_layout_tile_si32 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<32x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#ttnn_layout_tile_bf16 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<32x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module attributes {} {
  func.func @concat_with_tile_i32_non_divisible_shapes(%arg0: tensor<1x1x17x32xsi32, #ttnn_layout_tile_si32>, %arg1: tensor<1x1x15x32xsi32, #ttnn_layout_tile_si32>) -> tensor<1x1x32x32xbf16, #ttnn_layout_tile_bf16> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device

    // The op validation pass should detect that concat op will return si32 output,
    // but the output layout doesn't match the expected one, so it inserts a revert to tile layout with bf16.

    // CHECK: %[[CONCAT_RES:.*]] = "ttnn.concat"
    // CHECK: "ttnn.to_layout"
    // CHECK-SAME: (%[[CONCAT_RES]]
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>

    %1 = "ttnn.concat"(%arg0, %arg1) <{
      dim = 2 : si32
    }> : (tensor<1x1x17x32xsi32, #ttnn_layout_tile_si32>, tensor<1x1x15x32xsi32, #ttnn_layout_tile_si32>) -> tensor<1x1x32x32xbf16, #ttnn_layout_tile_bf16>

    return %1 : tensor<1x1x32x32xbf16, #ttnn_layout_tile_bf16>
  }
}
