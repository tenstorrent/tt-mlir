// REQUIRES: opmodel
// RUN: ttmlir-opt --ttcore-register-device --ttnn-operation-validation-and-fallback %s -o %t.mlir
// RUN: FileCheck %s --input-file %t.mlir

#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout_tile_input = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 2048 + d1 * 64 + d2, d3), <1x1>, memref<256x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout_tile_output = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 8192 + d1 * 128 + d2, d3), <1x1>, memref<1024x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module attributes {} {
  func.func @upsample_with_auto_validation(%arg0: tensor<4x32x64x3xbf16, #ttnn_layout_tile_input>) -> tensor<4x64x128x3xbf16, #ttnn_layout_tile_output> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device

    // The op validation pass should detect that upsample requires row-major input
    // and automatically insert layout conversion for tile input.
    // Also, the output layout of upsample op will be the same as input layout, so revert it back to the expected layout.

    // CHECK: "ttnn.to_layout"
    // CHECK-SAME: layout = #ttnn.layout<row_major>
    // CHECK-SAME: memory_config = #ttnn.memory_config<#dram, <interleaved>>
    // CHECK-NEXT: "ttnn.upsample"
    // CHECK-NEXT: "ttnn.to_layout"
    // CHECK-SAME: layout = #ttnn.layout<tile>

    %1 = "ttnn.upsample"(%arg0) <{mode = "nearest", scale_factor = 2 : si32}> : (tensor<4x32x64x3xbf16, #ttnn_layout_tile_input>) -> tensor<4x64x128x3xbf16, #ttnn_layout_tile_output>

    return %1 : tensor<4x64x128x3xbf16, #ttnn_layout_tile_output>
  }
}
