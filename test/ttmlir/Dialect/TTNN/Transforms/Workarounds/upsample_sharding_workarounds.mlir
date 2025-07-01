// RUN: ttmlir-opt --ttcore-register-device --ttnn-workaround --canonicalize %s | FileCheck %s

#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>

#layout_tile_interleaved = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<2048x30xf32, #dram>, <interleaved>>
#layout_tile_interleaved_out = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 2048 + d1 * 64 + d2, d3), <1x1>, memref<4096x30xf32, #dram>, <interleaved>>

#layout_rowmajor_interleaved = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<2048x30xbf16, #dram>, <interleaved>>
#layout_rowmajor_interleaved_out = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 2048 + d1 * 64 + d2, d3), <1x1>, memref<4096x30xbf16, #dram>, <interleaved>>

module {
  func.func @test_layout_and_sharding(%arg0: tensor<2x32x32x30xf32, #layout_tile_interleaved>) -> tensor<2x64x64x30xf32, #layout_tile_interleaved_out> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: %[[TO_ROWMAJOR:.*]] = "ttnn.to_layout"(%arg0)
    // CHECK-SAME: layout = #ttnn.layout<row_major>
    // CHECK-SAME: dtype = #tt.supportedDataTypes<bf16>
    // CHECK: %[[TO_SHARDED:.*]] = "ttnn.to_memory_config"(%[[TO_ROWMAJOR]])
    // CHECK-SAME: memory_config = #ttnn.memory_config<#ttnn.buffer_type<l1>, <height_sharded>
    // CHECK: %[[UPSAMPLE:.*]] = "ttnn.upsample"(%[[TO_SHARDED]])
    // CHECK-SAME: mode = "bilinear"
    // CHECK: %[[TO_INTERLEAVED:.*]] = "ttnn.to_memory_config"(%[[UPSAMPLE]])
    // CHECK-SAME: memory_config = #ttnn.memory_config<#ttnn.buffer_type<dram>, <interleaved>>
    // CHECK: %[[TO_TILE:.*]] = "ttnn.to_layout"(%[[TO_INTERLEAVED]])
    // CHECK-SAME: layout = #ttnn.layout<tile>
    // CHECK-SAME: dtype = #tt.supportedDataTypes<f32>
    %1 = "ttnn.upsample"(%arg0) <{mode = "bilinear", scale_factor = 2 : si32}> : (tensor<2x32x32x30xf32, #layout_tile_interleaved>) -> tensor<2x64x64x30xf32, #layout_tile_interleaved_out>
    return %1 : tensor<2x64x64x30xf32, #layout_tile_interleaved_out>
  }

  func.func @test_padding_and_sharding(%arg0: tensor<1x16x16x30xbf16, #layout_rowmajor_interleaved>) -> tensor<1x32x32x30xbf16, #layout_rowmajor_interleaved_out> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: %[[PADDED:.*]] = "ttnn.pad"(%arg0)
    // CHECK-SAME: padding = [0, 0, 0, 0, 0, 0, 0, 2]
    // CHECK: %[[TO_SHARDED:.*]] = "ttnn.to_memory_config"(%[[PADDED]])
    // CHECK-SAME: memory_config = #ttnn.memory_config<#ttnn.buffer_type<l1>, <height_sharded>
    // CHECK: %[[UPSAMPLE:.*]] = "ttnn.upsample"(%[[TO_SHARDED]])
    // CHECK-SAME: mode = "bilinear"
    // CHECK: %[[TO_INTERLEAVED:.*]] = "ttnn.to_memory_config"(%[[UPSAMPLE]])
    // CHECK-SAME: memory_config = #ttnn.memory_config<#ttnn.buffer_type<dram>, <interleaved>>
    // CHECK: %[[SLICED:.*]] = "ttnn.slice"(%[[TO_INTERLEAVED]])
    // CHECK-SAME: begins = [0, 0, 0, 0]
    // CHECK-SAME: ends = [1, 32, 32, 30]
    %1 = "ttnn.upsample"(%arg0) <{mode = "bilinear", scale_factor = 2 : si32}> : (tensor<1x16x16x30xbf16, #layout_rowmajor_interleaved>) -> tensor<1x32x32x30xbf16, #layout_rowmajor_interleaved_out>
    return %1 : tensor<1x32x32x30xbf16, #layout_rowmajor_interleaved_out>
  }

  func.func @test_layout_padding_sharding(%arg0: tensor<1x8x8x30xf32, #layout_tile_interleaved>) -> tensor<1x16x16x30xf32, #layout_tile_interleaved_out> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: %[[TO_ROWMAJOR:.*]] = "ttnn.to_layout"(%arg0)
    // CHECK-SAME: layout = #ttnn.layout<row_major>
    // CHECK-SAME: dtype = #tt.supportedDataTypes<bf16>
    // CHECK: %[[PADDED:.*]] = "ttnn.pad"(%[[TO_ROWMAJOR]])
    // CHECK-SAME: padding = [0, 0, 0, 0, 0, 0, 0, 2]
    // CHECK: %[[TO_SHARDED:.*]] = "ttnn.to_memory_config"(%[[PADDED]])
    // CHECK-SAME: memory_config = #ttnn.memory_config<#ttnn.buffer_type<l1>, <height_sharded>
    // CHECK: %[[UPSAMPLE:.*]] = "ttnn.upsample"(%[[TO_SHARDED]])
    // CHECK-SAME: mode = "bilinear"
    // CHECK: %[[TO_INTERLEAVED:.*]] = "ttnn.to_memory_config"(%[[UPSAMPLE]])
    // CHECK-SAME: memory_config = #ttnn.memory_config<#ttnn.buffer_type<dram>, <interleaved>>
    // CHECK: %[[SLICED:.*]] = "ttnn.slice"(%[[TO_INTERLEAVED]])
    // CHECK-SAME: begins = [0, 0, 0, 0]
    // CHECK-SAME: ends = [1, 16, 16, 30]
    // CHECK: %[[TO_TILE:.*]] = "ttnn.to_layout"(%[[SLICED]])
    // CHECK-SAME: layout = #ttnn.layout<tile>
    // CHECK-SAME: dtype = #tt.supportedDataTypes<f32>
    %1 = "ttnn.upsample"(%arg0) <{mode = "bilinear", scale_factor = 2 : si32}> : (tensor<1x8x8x30xf32, #layout_tile_interleaved>) -> tensor<1x16x16x30xf32, #layout_tile_interleaved_out>
    return %1 : tensor<1x16x16x30xf32, #layout_tile_interleaved_out>
  }

  func.func @test_no_workarounds_needed(%arg0: tensor<1x16x16x32xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 16 + d2, d3), <16x1>, memref<16x32xbf16, #l1>, <height_sharded>>>) -> tensor<1x32x32x32xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <16x1>, memref<64x32xbf16, #l1>, <height_sharded>>> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    // CHECK: %[[UPSAMPLE:.*]] = "ttnn.upsample"(%arg0)
    // CHECK-SAME: mode = "bilinear"
    // CHECK-NOT: "ttnn.to_layout"
    // CHECK-NOT: "ttnn.pad"
    // CHECK-NOT: "ttnn.slice"
    // CHECK-NOT: "ttnn.to_memory_config"
    %1 = "ttnn.upsample"(%arg0) <{mode = "bilinear", scale_factor = 2 : si32}> : (tensor<1x16x16x32xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 16 + d2, d3), <16x1>, memref<16x32xbf16, #l1>, <height_sharded>>>) -> tensor<1x32x32x32xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <16x1>, memref<64x32xbf16, #l1>, <height_sharded>>>
    return %1 : tensor<1x32x32x32xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <16x1>, memref<64x32xbf16, #l1>, <height_sharded>>>
  }
}
