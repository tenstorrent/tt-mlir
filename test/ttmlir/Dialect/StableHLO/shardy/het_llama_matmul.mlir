sdy.mesh @mesh = <["model"=8, "batch"=4]>

func.func @from_llama(
    %arg0: tensor<544x8192xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}]>},
    %arg1: tensor<8192x1024xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {"model"}]>})
    -> (tensor<544x1024xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {"model"}]>}) {
  %0 = stablehlo.dot_general %arg0, %arg1, contracting_dims = [1] x [0] {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"batch"}, {"model"}]>]>} : (tensor<544x8192xbf16>, tensor<8192x1024xbf16>) -> tensor<544x1024xbf16>
  return %0 : tensor<544x1024xbf16>
}
