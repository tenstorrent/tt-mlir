// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-pipeline %s

module {
  sdy.mesh @mesh = <["_axis_0"=8]>

  func.func @gather_embedding_lookup_with_all_slice(
      %arg0: tensor<201088x2880xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
      %arg1: tensor<1x128xi64> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>})
      -> (tensor<128x2880xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"_axis_0"}]>}) {
    %0 = stablehlo.reshape %arg1 : (tensor<1x128xi64>) -> tensor<128xi64>
    %1 = stablehlo.convert %0 : (tensor<128xi64>) -> tensor<128xui32>
    %2 = sdy.all_slice [{}, {"_axis_0"}] %arg0 out_sharding=<@mesh, [{}, {"_axis_0"}]> : tensor<201088x2880xbf16>
    %3 = "stablehlo.gather"(%2, %1) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1, 2880>}> : (tensor<201088x2880xbf16>, tensor<128xui32>) -> tensor<128x2880xbf16>
    return %3 : tensor<128x2880xbf16>
  }
}
module {
  sdy.mesh @mesh = <["axis_0"=8]>

  func.func @gather_embedding_lookup_sharded_embedding_dim(
      %arg0: tensor<201088x2880xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"axis_0"}]>},
      %arg1: tensor<128xui32> {sdy.sharding = #sdy.sharding<@mesh, [{}]>})
      -> (tensor<128x2880xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"axis_0"}]>}) {

    // After sharding propagation, %arg0 will be locally: tensor<201088x360xbf16>
    // The gather should still work, but slice_sizes needs to be updated to [1, 360]

    %0 = "stablehlo.gather"(%arg0, %arg1) <{
      dimension_numbers = #stablehlo.gather<
        offset_dims = [1],
        collapsed_slice_dims = [0],
        start_index_map = [0],
        index_vector_dim = 1
      >,
      indices_are_sorted = false,
      slice_sizes = array<i64: 1, 2880>
    }> : (tensor<201088x2880xbf16>, tensor<128xui32>) -> tensor<128x2880xbf16>

    return %0 : tensor<128x2880xbf16>
  }
}
