// RUN: ttmlir-opt --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t
#dram = #ttnn.buffer_type<dram>
#ttnn_layout_device_tile_f32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout_device_tile_si32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#ttnn_layout_device_tile_bf16 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
module {
  // Test case to verify the folding of reshape operation.
  func.func @identity_reshape(%arg0: tensor<64x128xf32, #ttnn_layout_device_tile_bf16>) -> tensor<64x128xf32, #ttnn_layout_device_tile_bf16> {
    // Verify that we fold the reshape when we reshape to the identical shape.
    // CHECK-NOT: ttnn.reshape
    // CHECK: return %arg0 : tensor<64x128xf32, #ttnn_layout>
    %0 = "ttnn.reshape"(%arg0) <{shape = [64 : i32, 128 : i32]}> : (tensor<64x128xf32, #ttnn_layout_device_tile_bf16>) -> tensor<64x128xf32, #ttnn_layout_device_tile_bf16>
    return %0 : tensor<64x128xf32, #ttnn_layout_device_tile_bf16>
  }

  // Test case to verify consecutive reshape op folding.
  func.func @consecutive_reshape(%arg0: tensor<64x128xf32, #ttnn_layout_device_tile_f32>) -> tensor<8192xf32, #ttnn_layout_device_tile_f32> {
    // Verify that we fold two consecutive reshape ops into a single one.
    // CHECK: ttnn.reshape
    // CHECK-NOT: ttnn.reshape
    // CHECK-NEXT: return
    %0 = "ttnn.reshape"(%arg0) <{shape = [32 : i32, 256 : i32]}> : (tensor<64x128xf32, #ttnn_layout_device_tile_f32>) -> tensor<32x256xf32, #ttnn_layout_device_tile_f32>
    %1 = "ttnn.reshape"(%0) <{shape = [8192 : i32]}> : (tensor<32x256xf32, #ttnn_layout_device_tile_f32>) -> tensor<8192xf32, #ttnn_layout_device_tile_f32>
    return %1 : tensor<8192xf32, #ttnn_layout_device_tile_f32>
  }

  // Test case to verify that we do not fold consecutive reshape ops if the first reshape has more than a single use.
  func.func @consecutive_reshape_multiple_uses(%arg0: tensor<64x128xf32, #ttnn_layout_device_tile_f32>) -> (tensor<8192xf32, #ttnn_layout_device_tile_f32>, tensor<32x256xf32, #ttnn_layout_device_tile_f32>) {
    // Verify that both reshapes exists.
    // CHECK: ttnn.reshape
    // CHECK: ttnn.reshape
    // CHECK: ttnn.add
    %0 = "ttnn.reshape"(%arg0) <{shape = [32 : i32, 256 : i32]}> : (tensor<64x128xf32, #ttnn_layout_device_tile_f32>) -> tensor<32x256xf32, #ttnn_layout_device_tile_f32>
    %1 = "ttnn.reshape"(%0) <{shape = [8192 : i32]}> : (tensor<32x256xf32, #ttnn_layout_device_tile_f32>) -> tensor<8192xf32, #ttnn_layout_device_tile_f32>
    %2 = "ttnn.add"(%0, %0) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<32x256xf32, #ttnn_layout_device_tile_f32>, tensor<32x256xf32, #ttnn_layout_device_tile_f32>) -> tensor<32x256xf32, #ttnn_layout_device_tile_f32>
    return %1, %2 : tensor<8192xf32, #ttnn_layout_device_tile_f32>, tensor<32x256xf32, #ttnn_layout_device_tile_f32>
  }

  // Test case to verify that we do not fold seemingly consecutive reshapes if the result of the first reshape is consumed by an op which may modify the tensor in place BEFORE the second reshape consumes the same tensor.
  func.func @consecutive_reshape_multiple_uses_modify_in_place(%arg0: tensor<1x1x12x32x64xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3, d4) -> (d0 * 384 + d1 * 384 + d2 * 32 + d3, d4), <1x1>, memref<12x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>, %arg1: tensor<1x12x1x64xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 384 + d1 * 32 + d2, d3), <1x1>, memref<12x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>, %arg2: tensor<1xsi32, #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1x!ttcore.tile<32x32, si32>, #ttnn.buffer_type<dram>>, <interleaved>>>) -> tensor<1x1x12x32x64xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3, d4) -> (d0 * 384 + d1 * 384 + d2 * 32 + d3, d4), <1x1>, memref<12x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>> {
    // Verify that both reshapes exists.
    // CHECK: "ttnn.reshape"(%arg0
    // CHECK: "ttnn.update_cache"(%0
    // CHECK: "ttnn.reshape"(%0
    %0 = "ttnn.reshape"(%arg0) <{shape = [1 : i32, 12 : i32, 32 : i32, 64 : i32]}> : (tensor<1x1x12x32x64xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3, d4) -> (d0 * 384 + d1 * 384 + d2 * 32 + d3, d4), <1x1>, memref<12x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>) -> tensor<1x12x32x64xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 384 + d1 * 32 + d2, d3), <1x1>, memref<12x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>
    "ttnn.update_cache"(%0, %arg1, %arg2) <{batch_offset = 0 : i32}> : (tensor<1x12x32x64xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 384 + d1 * 32 + d2, d3), <1x1>, memref<12x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>, tensor<1x12x1x64xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 384 + d1 * 32 + d2, d3), <1x1>, memref<12x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>, tensor<1xsi32, #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x1x!ttcore.tile<32x32, si32>, #ttnn.buffer_type<dram>>, <interleaved>>>) -> ()
    %1 = "ttnn.reshape"(%0) <{shape = [1 : i32, 1 : i32, 12 : i32, 32 : i32, 64 : i32]}> : (tensor<1x12x32x64xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 384 + d1 * 32 + d2, d3), <1x1>, memref<12x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>) -> tensor<1x1x12x32x64xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3, d4) -> (d0 * 384 + d1 * 384 + d2 * 32 + d3, d4), <1x1>, memref<12x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>
    return %1 : tensor<1x1x12x32x64xbf16, #ttnn.ttnn_layout<(d0, d1, d2, d3, d4) -> (d0 * 384 + d1 * 384 + d2 * 32 + d3, d4), <1x1>, memref<12x2x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<dram>>, <interleaved>>>
  }
}
