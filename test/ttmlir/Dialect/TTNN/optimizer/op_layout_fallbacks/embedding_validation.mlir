// REQUIRES: opmodel
// RUN: ttmlir-opt --ttcore-register-device --ttnn-operation-validation-and-fallback %s -o %t.mlir
// RUN: FileCheck %s --input-file %t.mlir

#dram = #ttnn.buffer_type<dram>
#ttnn_layout_tile_ui32 = #ttnn.ttnn_layout<(d0, d1) -> (d0 * 32 + d1), <1x1>, memref<32x1x!ttcore.tile<32x32, u32>, #dram>, <interleaved>>
#ttnn_layout_tile_f32 = #ttnn.ttnn_layout<(d0, d1) -> (d0 * 32 + d1), <1x1>, memref<32x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout_row_major_bf16 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 10 + d1, d2), <1x1>, memref<320x32xbf16, #ttnn.buffer_type<dram>>, <interleaved>>
#ttnn_layout_tile_input = #ttnn.ttnn_layout<(d0, d1) -> (d0 * 32 + d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout_tile_output = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<32x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module attributes {} {
  func.func @embedding_with_row_major_output(%arg0: tensor<32x10xui32, #ttnn_layout_tile_ui32>, %arg1: tensor<100x32xf32, #ttnn_layout_tile_f32>) -> tensor<32x10x32xbf16, #ttnn_layout_row_major_bf16> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device

    // The post-optimizer validation should detect that embedding requires:
    // BFloat16 data type for weight tensor (f32 -> bf16)

    // CHECK: %[[CONVERTED:.*]] = "ttnn.to_layout"
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK: "ttnn.embedding"
    // CHECK-SAME: %[[CONVERTED]])

    %1 = "ttnn.embedding"(%arg0, %arg1) : (tensor<32x10xui32, #ttnn_layout_tile_ui32>, tensor<100x32xf32, #ttnn_layout_tile_f32>) -> tensor<32x10x32xbf16, #ttnn_layout_row_major_bf16>

    return %1 : tensor<32x10x32xbf16, #ttnn_layout_row_major_bf16>
  }

  func.func @embedding_with_tile_output(%arg0: tensor<32x32xf32, #ttnn_layout_tile_input>, %arg1: tensor<1000x32xf32, #ttnn_layout_tile_f32>) -> tensor<32x32x32xbf16, #ttnn_layout_tile_output> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device

    // The post-optimizer validation should detect that embedding requires:
    // BFloat16 data type for weight tensor (f32 -> bf16)

    // CHECK: %[[ARG0_BF16:.*]] = "ttnn.to_layout"
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK: %[[ARG1_BF16:.*]] = "ttnn.to_layout"
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK: "ttnn.embedding"
    // CHECK-SAME: (%[[ARG0_BF16]], %[[ARG1_BF16]])

    %1 = "ttnn.embedding"(%arg0, %arg1) : (tensor<32x32xf32, #ttnn_layout_tile_input>, tensor<1000x32xf32, #ttnn_layout_tile_f32>) -> tensor<32x32x32xbf16, #ttnn_layout_tile_output>

    return %1 : tensor<32x32x32xbf16, #ttnn_layout_tile_output>
  }
}
