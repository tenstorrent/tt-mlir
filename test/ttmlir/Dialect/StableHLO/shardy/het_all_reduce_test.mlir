func.func @dot_compatible_contracting_dim_sharded(
    %arg0: tensor<8x32xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {"model"}]>},
    %arg1: tensor<32x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"model"}, {}]>})
    -> (tensor<8x16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}]>}) {
  %0 = stablehlo.dot %arg0, %arg1 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"batch"}, {}]>]>} : (tensor<8x32xf32>, tensor<32x16xf32>) -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}
