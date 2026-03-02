#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<1x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
module attributes {ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x2>]>} {
  func.func @forward(%arg0: tensor<1x32x64xbf16, #ttnn_layout>) -> tensor<1x32x64xbf16, #ttnn_layout> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x2>}> : () -> !ttnn.device
    %1 = "ttnn.mesh_shard"(%arg0, %0) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<1x32x64xbf16, #ttnn_layout>, !ttnn.device) -> tensor<1x32x64xbf16, #ttnn_layout>
    %2 = "ttnn.mesh_shard"(%1, %0) <{shard_dims = array<i64: -1>, shard_direction = #ttcore.shard_direction<shard_to_full>, shard_shape = array<i64: 1>, shard_type = #ttcore.shard_type<identity>}> : (tensor<1x32x64xbf16, #ttnn_layout>, !ttnn.device) -> tensor<1x32x64xbf16, #ttnn_layout>
    return %2 : tensor<1x32x64xbf16, #ttnn_layout>
  }
}
