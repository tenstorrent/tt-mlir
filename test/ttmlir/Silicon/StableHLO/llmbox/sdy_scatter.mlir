// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-pipeline="mesh-shape=1,8 automatic-arg-analysis" --stablehlo-to-ttir-pipeline --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% mesh-shape=1,8" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir

module {
  // Test scatter with sharding along scatter_dims_to_operand_dims.
  sdy.mesh @mesh = <["model"=1, "batch"=8]>
  func.func @scatter_test_replicate_input(%arg0: tensor<71x4xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
                          %arg1: tensor<71x4x2xi64> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}]>},
                          %arg2: tensor<71x32xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"batch"}]>}) -> tensor<71x32xbf16> {
    %0 = "stablehlo.scatter"(%arg2, %arg1, %arg0) <{scatter_dimension_numbers = #stablehlo.scatter<inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 2>}> ({
    ^bb0(%arg3: tensor<bf16>, %arg4: tensor<bf16>):
      stablehlo.return %arg4 : tensor<bf16>
    }) : (tensor<71x32xbf16>, tensor<71x4x2xi64>, tensor<71x4xbf16>) -> tensor<71x32xbf16>
    return %0 : tensor<71x32xbf16>
  }
  // Test scatter with sharding NOT along scatter_dims_to_operand_dims.
  func.func @scatter_test_shard_input(%arg0: tensor<4x8xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
                          %arg1: tensor<4x1xi64> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>},
                          %arg2: tensor<32x64xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}, {"batch"}]>}) -> tensor<32x64xbf16> {
    %0 = "stablehlo.scatter"(%arg2, %arg1, %arg0) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>}> ({
    ^bb0(%arg3: tensor<bf16>, %arg4: tensor<bf16>):
      stablehlo.return %arg4 : tensor<bf16>
    }) : (tensor<32x64xbf16>, tensor<4x1xi64>, tensor<4x8xbf16>) -> tensor<32x64xbf16>
    return %0 : tensor<32x64xbf16>
  }
}
