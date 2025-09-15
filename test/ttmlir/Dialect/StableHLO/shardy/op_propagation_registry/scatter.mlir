sdy.mesh @mesh = <["model"=1, "batch"=8]>

func.func @scatter_test(%operand: tensor<13x32xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"batch"}]>}, %indices: tensor<13x4x2xi64>, %updates: tensor<13x4xbf16>) -> tensor<13x32xbf16> {
  %0 = "stablehlo.scatter"(
        %operand,        
        %indices,        
        %updates
      ) <{
        scatter_dimension_numbers = #stablehlo.scatter<
          inserted_window_dims = [0, 1],
          scatter_dims_to_operand_dims = [0, 1],
          index_vector_dim = 2
        >
      }> ({
        ^bb0(%arg68: tensor<bf16>, %arg69: tensor<bf16>):
          "stablehlo.return"(%arg69) : (tensor<bf16>) -> ()
      }) {
        sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{?}, {"batch", ?}]>]>
      } : (
        tensor<13x32xbf16>,
        tensor<13x4x2xi64>,
        tensor<13x4xbf16>
      ) -> tensor<13x32xbf16>
  return %0 : tensor<13x32xbf16>
}

module {
  sdy.mesh @mesh = <["model"=1, "batch"=8]>
  func.func @scatter_test(%arg0: tensor<13x32xbf16> {ttcore.shard_status = #ttcore.shard_status<presharded>}, %arg1: tensor<13x4x2xi64> {ttcore.shard_status = #ttcore.shard_status<unsharded>}, %arg2: tensor<13x4xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) -> (tensor<13x32xbf16> {ttcore.shard_status = #ttcore.shard_status<unsharded>}) {
    %0 = sdy.manual_computation(%arg0, %arg1, %arg2) in_shardings=[<@mesh, [{}, {"batch"}]>, <@mesh, [{}, {}, {}]>, <@mesh, [{}, {}]>] out_shardings=[<@mesh, [{}, {"batch"}]>] manual_axes={"model", "batch"} (%arg3: tensor<13x4xbf16>, %arg4: tensor<13x4x2xi64>, %arg5: tensor<13x4xbf16>) {
      %1 = "stablehlo.scatter"(%arg3, %arg4, %arg5) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 2>}> ({
      ^bb0(%arg6: tensor<bf16>, %arg7: tensor<bf16>):
        stablehlo.return %arg7 : tensor<bf16>
      }) : (tensor<13x4xbf16>, tensor<13x4x2xi64>, tensor<13x4xbf16>) -> tensor<13x4xbf16>
      sdy.return %1 : tensor<13x4xbf16>
    } : (tensor<13x32xbf16>, tensor<13x4x2xi64>, tensor<13x4xbf16>) -> tensor<13x32xbf16>
    return %0 : tensor<13x32xbf16>
  }
}