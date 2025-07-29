// RUN: ttmlir-opt --ttcore-register-device --ttnn-workaround --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>

#layout_tile_interleaved = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 32 + d2, d3), <1x1>, memref<8x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#layout_tile_interleaved_out = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 32 + d2, d3), <1x1>, memref<16x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

module {
  func.func @test_layout_padding_sharding(%arg0: tensor<1x8x8x30xf32, #layout_tile_interleaved>) -> tensor<1x16x16x30xf32, #layout_tile_interleaved_out> {
    // CHECK: %[[PADDED:.*]] = "ttnn.pad"(%arg0)
    // CHECK-SAME: padding = array<i32: 0, 0, 0, 0, 0, 0, 0, 2>
    // CHECK: %[[TO_ROWMAJOR:.*]] = "ttnn.to_layout"(%[[PADDED]])
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK-SAME: layout = #ttnn.layout<row_major>
    // CHECK: %[[UPSAMPLE:.*]] = "ttnn.upsample"(%[[TO_ROWMAJOR]])
    // CHECK-SAME: mode = "bilinear"
    // CHECK: %[[TO_TILE:.*]] = "ttnn.to_layout"(%[[UPSAMPLE]])
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<f32>
    // CHECK-SAME: layout = #ttnn.layout<tile>
    // CHECK: "ttnn.slice"(%[[TO_TILE]])
    // CHECK-SAME: begins = [0 : i32, 0 : i32, 0 : i32, 0 : i32]
    // CHECK-SAME: ends = [1 : i32, 16 : i32, 16 : i32, 30 : i32]
    %1 = "ttnn.upsample"(%arg0) <{mode = "bilinear", scale_factor = 2 : si32}> : (tensor<1x8x8x30xf32, #layout_tile_interleaved>) -> tensor<1x16x16x30xf32, #layout_tile_interleaved_out>
    return %1 : tensor<1x16x16x30xf32, #layout_tile_interleaved_out>
  }

  func.func @test_no_workarounds_needed(%arg0: tensor<1x16x16x32xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 16 + d2, d3), <16x1>, memref<16x32xbf16, #l1>, <height_sharded>>>) -> tensor<1x32x32x32xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <16x1>, memref<64x32xbf16, #l1>, <height_sharded>>> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: %[[UPSAMPLE:.*]] = "ttnn.upsample"(%arg0)
    // CHECK-SAME: mode = "bilinear"
    // CHECK-NOT: "ttnn.to_layout"
    // CHECK-NOT: "ttnn.pad"
    %1 = "ttnn.upsample"(%arg0) <{mode = "bilinear", scale_factor = 2 : si32}> : (tensor<1x16x16x32xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 16 + d2, d3), <16x1>, memref<16x32xbf16, #l1>, <height_sharded>>>) -> tensor<1x32x32x32xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <16x1>, memref<64x32xbf16, #l1>, <height_sharded>>>
    return %1 : tensor<1x32x32x32xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <16x1>, memref<64x32xbf16, #l1>, <height_sharded>>>
  }
}
