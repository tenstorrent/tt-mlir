#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1 + d1, d2 * 8 + d3), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1 + d1, d2 * 32 + d3), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1 + d1, d2 * 128 + d3), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
module attributes {ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x4>]>} {
  func.func @forward(%arg0: tensor<1x1x32x128xbf16, #ttnn_layout2>) -> tensor<1x1x32x32xbf16, #ttnn_layout1> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x4>}> : () -> !ttnn.device
    %1 = "ttnn.distribute_tensor"(%arg0, %0) <{mapper_config = #ttnn.mesh_mapper_config<placements = [<shard, 3 : i64>]>}> : (tensor<1x1x32x128xbf16, #ttnn_layout2>, !ttnn.device) -> tensor<1x1x32x32xbf16, #ttnn_layout1>
    %2 = "ttnn.reduce_scatter"(%1) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>, scatter_dim = 3 : si32}> : (tensor<1x1x32x32xbf16, #ttnn_layout1>) -> tensor<1x1x32x8xbf16, #ttnn_layout>
    %3 = "ttnn.aggregate_tensor"(%2, %0) <{composer_config = #ttnn.mesh_composer_config<dims = [3 : i64]>}> : (tensor<1x1x32x8xbf16, #ttnn_layout>, !ttnn.device) -> tensor<1x1x32x32xbf16, #ttnn_layout1>
    return %3 : tensor<1x1x32x32xbf16, #ttnn_layout1>
  }
}
