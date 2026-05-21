// RUN: ttmlir-opt --ttnn-kv-cache-dtype-conversion="target-dtype=bfp_bf8" %s | FileCheck %s

#dram = #ttnn.buffer_type<dram>

#global_cache = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 2048 + d1 * 64 + d2, d3), <1x1>, memref<128x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#local_cache = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 2048 + d1 * 64 + d2, d3), <1x1>, memref<64x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#input = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 + d2, d3), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

// CHECK-LABEL: func.func @test_fill_cache_bfp8_tp
// CHECK-SAME: %arg0: tensor<2x32x64x128x!ttcore.tile<32x32, bfp_bf8>,
// CHECK-SAME: %arg1: tensor<1x32x1x128xbf16,
module attributes {} {
  func.func @test_fill_cache_bfp8_tp(
      %arg0: tensor<2x32x64x128xbf16, #global_cache> {ttcore.kv_cache},
      %arg1: tensor<1x32x1x128xbf16, #input>
  ) attributes {tt.function_type = "forward_device"} {
    %dev = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x2>}> : () -> !ttnn.device

    // CHECK: %[[LOCAL:.*]] = "ttnn.mesh_shard"(%arg0,
    // CHECK-SAME: -> tensor<1x32x64x128x!ttcore.tile<32x32, bfp_bf8>,
    %local_cache = "ttnn.mesh_shard"(%arg0, %dev) <{shard_dims = array<i64: -1, 0>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 2, 1, 1, 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<2x32x64x128xbf16, #global_cache>, !ttnn.device) -> tensor<1x32x64x128xbf16, #local_cache>

    // CHECK: %[[CAST:.*]] = "ttnn.typecast"(%arg1)
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bfp_bf8>
    // CHECK-SAME: -> tensor<1x32x1x128x!ttcore.tile<32x32, bfp_bf8>,
    // CHECK: "ttnn.fill_cache"(%[[LOCAL]], %[[CAST]])
    // CHECK-NOT: "ttnn.typecast"(%arg0)
    "ttnn.fill_cache"(%local_cache, %arg1) <{batch_offset = 0 : i32}> : (
        tensor<1x32x64x128xbf16, #local_cache>,
        tensor<1x32x1x128xbf16, #input>
    ) -> ()
    return
  }
}
