// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-pipeline -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

sdy.mesh @mesh = <["model"=1, "batch"=2]>

func.func @gather_with_shard_on_operand_non_index_dim(%arg0: tensor<4x56x56x96xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}, {}, {}]>}, %arg1: tensor<56xi64>) -> tensor<4x56x56x96xbf16> {
  %0 = "stablehlo.gather"(%arg0, %arg1) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 4, 1, 56, 96>}> : (tensor<4x56x56x96xbf16>, tensor<56xi64>) -> tensor<4x56x56x96xbf16>
  return %0 : tensor<4x56x56x96xbf16>
}

// CHECK: sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh, [{"batch"}, {}, {}, {}]>, <@mesh, [{}]>] out_shardings=[<@mesh, [{"batch"}, {}, {}, {}]>]
// CHECK: "stablehlo.gather"(%arg2, %arg3) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 2, 1, 56, 96>}> : (tensor<2x56x56x96xbf16>, tensor<56xi64>) -> tensor<2x56x56x96xbf16>
// CHECK: sdy.return %1 : tensor<2x56x56x96xbf16>


func.func @gather_with_shard_on_operand_index_dim(%arg0: tensor<4x56x56x96xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}, {}, {}]>}, %arg1: tensor<56x2xi64>) -> tensor<1x56x56x96xbf16> {
  %0 = "stablehlo.gather"(%arg0, %arg1) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 56, 96>}> : (tensor<4x56x56x96xbf16>, tensor<56x2xi64>) -> tensor<1x56x56x96xbf16>
  return %0 : tensor<1x56x56x96xbf16>
}

// CHECK: sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh, [{"batch"}, {}, {}, {}]>, <@mesh, [{}, {}]>] out_shardings=[<@mesh, [{}, {}, {}, {}]>]
// CHECK: "stablehlo.gather"(%arg2, %arg3) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [0, 1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 1, 1, 56, 96>}> : (tensor<2x56x56x96xbf16>, tensor<56x2xi64>) -> tensor<1x56x56x96xbf16>
// CHECK: %2 = "stablehlo.all_reduce"(%1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense
// CHECK-SAME: 0, 1
// CHECK: %3 = stablehlo.add %arg4, %arg5 : tensor<bf16>
// CHECK: stablehlo.return %3 : tensor<bf16>
// CHECK: sdy.return %2 : tensor<1x56x56x96xbf16>


func.func @gather_with_shard_on_start_indices_0(%arg0: tensor<4x56x56x96xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}, {}, {}]>}, %arg1: tensor<56xi64> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}]>}) -> tensor<4x56x56x96xbf16> {
  %0 = "stablehlo.gather"(%arg0, %arg1) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 4, 1, 56, 96>}> : (tensor<4x56x56x96xbf16>, tensor<56xi64>) -> tensor<4x56x56x96xbf16>
  return %0 : tensor<4x56x56x96xbf16>
}

// CHECK: sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh, [{}, {}, {}, {}]>, <@mesh, [{"batch"}]>] out_shardings=[<@mesh, [{}, {"batch"}, {}, {}]>]
// CHECK: "stablehlo.gather"(%arg2, %arg3) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 4, 1, 56, 96>}> : (tensor<4x56x56x96xbf16>, tensor<28xi64>) -> tensor<4x28x56x96xbf16>
// CHECK: sdy.return %1 : tensor<4x28x56x96xbf16>


func.func @gather_with_shard_on_start_indices_from_embedding_op(%arg0: tensor<131072x2048xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}]>}, %arg1: tensor<2x12xi64> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}]>}) -> tensor<2x12x2048xbf16> {
  %0 = "stablehlo.gather"(%arg0, %arg1) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 2048>}> : (tensor<131072x2048xbf16>, tensor<2x12xi64>) -> tensor<2x12x2048xbf16>
  return %0 : tensor<2x12x2048xbf16>
}

// CHECK: sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh, [{}, {}]>, <@mesh, [{"batch"}, {}]>] out_shardings=[<@mesh, [{"batch"}, {}, {}]>]
// CHECK: "stablehlo.gather"(%arg2, %arg3) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 2048>}> : (tensor<131072x2048xbf16>, tensor<1x12xi64>) -> tensor<1x12x2048xbf16>
// CHECK: sdy.return %1 : tensor<1x12x2048xbf16>

func.func @gather_with_shard_on_op_result(
    %arg0: tensor<4x56x56x96xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}, {}, {}]>},
    %arg0_1: tensor<4x56x56x96xbf16> {sdy.sharding = #sdy.sharding<@mesh, [{"batch"}, {}, {}, {}]>},
    %arg1: tensor<56xi64>
) -> tensor<4x56x56x96xbf16> {
  %added = stablehlo.add %arg0, %arg0_1
      {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"batch"}, {}, {}, {}]>]>}
      : (tensor<4x56x56x96xbf16>, tensor<4x56x56x96xbf16>) -> tensor<4x56x56x96xbf16>
  %0 = "stablehlo.gather"(%added, %arg1) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 4, 1, 56, 96>}> : (tensor<4x56x56x96xbf16>, tensor<56xi64>) -> tensor<4x56x56x96xbf16>
  return %0 : tensor<4x56x56x96xbf16>
}

// CHECK: sdy.manual_computation(%arg0, %arg1, %arg2) in_shardings=[<@mesh, [{"batch"}, {}, {}, {}]>, <@mesh, [{"batch"}, {}, {}, {}]>, <@mesh, [{}]>] out_shardings=[<@mesh, [{"batch"}, {}, {}, {}]>]
// CHECK: "stablehlo.gather"(%1, %arg5) <{dimension_numbers = #stablehlo.gather<offset_dims = [0, 2, 3], collapsed_slice_dims = [1], start_index_map = [1], index_vector_dim = 1>, indices_are_sorted = false, slice_sizes = array<i64: 2, 1, 56, 96>}> : (tensor<2x56x56x96xbf16>, tensor<56xi64>) -> tensor<2x56x56x96xbf16>
// CHECK: sdy.return %2 : tensor<2x56x56x96xbf16>
